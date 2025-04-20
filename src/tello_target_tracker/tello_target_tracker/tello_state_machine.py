#!/usr/bin/env python3

import enum
import time
import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, Twist, Vector3
from tello_interfaces.msg import TargetInfo, DroneStatus

from tello_target_tracker.camera_getter import TelloCamera
from tello_target_tracker.person_tracker import PersonTracker
from std_srvs.srv import SetBool

# Import the ROS visualization module
from tello_target_tracker.ros_visualization import RosVisualizer, draw_tracking_status

# Import the Tello library
from tello_target_tracker.DJITelloPy.djitellopy import Tello

class DroneState(enum.Enum):
    IDLE = 0
    TAKEOFF = 1
    SEARCHING = 2
    WAITING_FOR_ASSIGNMENT = 3
    TRACKING = 4
    LANDING = 5
    EMERGENCY = 6

class TelloStateMachine(Node):
    def __init__(self):
        super().__init__('tello_state_machine')
        
        # Initialize parameters
        self.drone_id = self.declare_parameter('drone_id', 0).value
        self.search_direction = 1  # 1 for clockwise, -1 for counter-clockwise
        self.target_lost_threshold = 5.0  # seconds
        
        # Initialize state variables
        self.current_state = DroneState.IDLE
        self.previous_state = None
        self.targets = {}  # Dictionary of detected targets
        self.assigned_target_id = None
        self.target_last_seen = 0
        self.target_height = 175 # Default height in cm

        # Dictionary to save velocity values to integrate
        self.velocity = {"x": None, "y": None, "z": None}
        self.position = {"x": 0, "y": 0, "z": 0}  # Estimated position (starting at 0, 0, 0)

        # Flag to track whether camera is initialized
        self.camera_initialized = False
        self.camera_error = False
        self.last_camera_retry = 0
        self.camera_retry_interval = 5.0  # seconds
        
        # Initialize Tello drone
        self.tello = Tello()
        self.person_tracker = PersonTracker(
            conf_thresh=0.4,
            color_filter=False  # Set to True if you want to filter by color
        )
        
        # Target acquisition parameters
        self.min_detections_for_tracking = 3  # Reduce from 5 to make tracking faster
        self.detection_persistence_count = 0
        self.potential_target_id = None
        
        # Initialize the ROS visualizer
        self.visualizer = RosVisualizer(self, history_length=200)

        self.connect_to_drone()
        
        # Create ROS publishers
        self.state_pub = self.create_publisher(
            DroneStatus, 
            f'/drone_{self.drone_id}/status', 
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/drone_{self.drone_id}/cmd_vel',
            10
        )
        
        # Create ROS subscribers
        self.target_assign_sub = self.create_subscription(
            String,
            f'/target_prioritization/assignments',
            self.target_assignment_callback,
            10
        )
        
        self.start_service = self.create_service(
            SetBool, 
            f'/drone_{self.drone_id}/start_mission', 
            self.start_mission_callback
        )
        self.abort_service = self.create_service(
            SetBool, 
            f'/drone_{self.drone_id}/abort_mission', 
            self.abort_mission_callback
        )
                
        # Create timers for state machine execution and status updates
        self.timer = self.create_timer(0.1, self.state_machine_callback)  # 10Hz
        self.status_timer = self.create_timer(1.0, self.publish_status)  # 1Hz
        
        self.get_logger().info(f'Tello State Machine initialized for drone {self.drone_id}')
    
    def connect_to_drone(self):
        """Connect to the Tello drone and initialize it with retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.get_logger().info(f"Attempting to connect to Tello drone (attempt {retry_count+1}/{max_retries})...")
                self.tello.connect()
                battery = self.tello.get_battery()
                self.get_logger().info(f"Connected to Tello drone. Battery: {battery}%")
                if battery < 20:
                    self.get_logger().warn(f"Low battery: {battery}%! Charge before extended flight.")
                
                # Initialize camera after successful connection
                self.tello_camera = TelloCamera(self.tello)
                
                return True
            except Exception as e:
                retry_count += 1
                self.get_logger().error(f"Failed to connect to Tello drone: {str(e)}")
                if retry_count < max_retries:
                    self.get_logger().info(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    self.get_logger().error("Maximum retries exceeded. Check if drone is powered on and Wi-Fi is connected.")
                    return False
    
    def initialize_camera(self):
        """Initialize the camera with retry mechanism"""
        if self.camera_initialized:
            return True
            
        # Check if enough time has passed since last retry
        current_time = time.time()
        if current_time - self.last_camera_retry < self.camera_retry_interval:
            return False
            
        self.last_camera_retry = current_time
        
        try:
            self.get_logger().info("Initializing camera...")
            # Start the video capture
            self.tello_camera.start_video_capture()
            
            # Start the visualizer
            self.visualizer.start()
            
            # Wait for camera to actually produce frames
            for attempt in range(5):
                time.sleep(0.5)
                frame = self.tello_camera.get_frame()
                if frame is not None:
                    self.get_logger().info(f"Camera initialized successfully! Frame shape: {frame.shape}")
                    self.camera_initialized = True
                    self.camera_error = False
                    return True
            
            self.get_logger().error("Camera initialization timed out - no frames received")
            self.camera_error = True
            return False
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera: {str(e)}")
            self.camera_error = True
            return False
    
    def target_assignment_callback(self, msg):
        """Process target assignment from prioritization module"""
        # Parse the assignment message (format: "drone_id:target_id")
        assignments = msg.data.split(';')
        
        for assignment in assignments:
            if not assignment:
                continue
                
            drone_id, target_id = assignment.split(':')
            
            if int(drone_id) == self.drone_id:
                self.assigned_target_id = int(target_id)
                self.get_logger().info(f"Drone {self.drone_id} assigned to target {self.assigned_target_id}")
                
                # If we were searching or waiting, now we should track
                if self.current_state in [DroneState.SEARCHING, DroneState.WAITING_FOR_ASSIGNMENT]:
                    self.change_state(DroneState.TRACKING)
    
    def start_mission_callback(self, request, response):
        """Service callback to start the mission"""
        if self.current_state == DroneState.IDLE:
            self.start_mission()
            response.success = True
            response.message = "Mission started"
        else:
            self.get_logger().warn("Cannot start mission: drone not in IDLE state")
            response.success = False
            response.message = "Drone not in IDLE state"
        return response

    def abort_mission_callback(self, request, response):
        """Service callback to abort the mission"""
        if self.current_state != DroneState.IDLE:
            self.abort_mission()
            response.success = True
            response.message = "Mission aborted"
        else:
            self.get_logger().warn("Cannot abort mission: drone already in IDLE state")
            response.success = False
            response.message = "Drone already in IDLE state"
        return response
        
    def publish_status(self):
        """Publish drone status information"""
        try:
            status_msg = DroneStatus()
            status_msg.drone_id = self.drone_id
            status_msg.state = self.current_state.name
            status_msg.battery = self.tello.get_battery()
            status_msg.target_id = -1 if self.assigned_target_id is None else self.assigned_target_id
            status_msg.height = float(self.tello.get_height()) / 100.0  # Convert cm to meters
            
            # Get velocity information
            vel = Vector3()
            vel.x = float(self.tello.get_speed_x()) / 100.0  # Convert to m/s
            vel.y = float(self.tello.get_speed_y()) / 100.0
            vel.z = float(self.tello.get_speed_z()) / 100.0
            status_msg.velocity = vel
            
            self.state_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {str(e)}")
    
    def change_state(self, new_state):
        """Change the current state with logging"""
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.get_logger().info(f"State changed: {self.previous_state.name} -> {self.current_state.name}")
            
            # If transitioning to a state that needs camera, make sure it's ready
            if new_state in [DroneState.SEARCHING, DroneState.TRACKING] and not self.camera_initialized:
                self.initialize_camera()
    
    def check_target_timeout(self):
        """Check if the tracked target has timed out and manage recovery phases"""
        if self.assigned_target_id is None:
            return False
            
        elapsed_time = time.time() - self.target_last_seen
        
        # Target still visible recently
        if elapsed_time < 1.0:
            return False
            
        # Short loss - hover in place and look around slightly
        elif elapsed_time < 3.0:
            self.get_logger().debug(f"Target {self.assigned_target_id} briefly lost, hovering")
            self.tello.send_rc_control(0, 0, 0, 10 * self.search_direction)
            
            # Update the visualization
            self.visualizer.update_control_values(0, 0, 0, 10 * self.search_direction, time.time())
            
            return False
            
        # Medium loss - start turning in place to find target
        elif elapsed_time < self.target_lost_threshold:
            self.get_logger().debug(f"Target {self.assigned_target_id} lost for {elapsed_time:.1f}s, searching in place")
            self.tello.send_rc_control(0, 0, 0, 20 * self.search_direction)
            
            # Update the visualization
            self.visualizer.update_control_values(0, 0, 0, 20 * self.search_direction, time.time())
            
            return False
            
        # Long loss - target officially lost, return to searching state
        else:
            self.get_logger().warn(f"Target {self.assigned_target_id} lost for {elapsed_time:.1f}s! Returning to search.")
            self.assigned_target_id = None
            self.potential_target_id = None
            self.detection_persistence_count = 0
            return True
    
    def execute_search_pattern(self):
        """Execute a search pattern to find targets"""
        # Use Tello's RC control for rotation search
        # Rotate in place to look for targets
        self.tello.send_rc_control(0, 0, 0, 30 * self.search_direction)
        
        # Update the visualization
        self.visualizer.update_control_values(0, 0, 0, 30 * self.search_direction, time.time())
        
        # Toggle search direction periodically using a proper timer
        current_time = time.time()
        if not hasattr(self, 'last_direction_change'):
            self.last_direction_change = current_time
        
        # Change direction every 10 seconds
        if current_time - self.last_direction_change >= 10.0:
            self.search_direction *= -1  # Reverse direction
            self.last_direction_change = current_time
            self.get_logger().info(f"Changing search direction to {self.search_direction}")
    
    def execute_tracking(self):
        """Execute tracking behavior based on heading and distance"""
        if self.assigned_target_id is None or self.assigned_target_id not in self.targets:
            return
            
        target = self.targets[self.assigned_target_id]
        heading = target['heading']
        distance = target['distance']
        height_offset = target['position'].y
        
        # X axis control (forward/backward)
        # Maintain target distance of 150cm
        target_distance = 150.0  # Desired distance in cm
        distance_error = distance - target_distance
        x_vel = int(max(min(distance_error * 0.2, 30), -30))
        
        # Y axis control (left/right)
        # Use heading to center the target
        y_vel = int(max(min(-heading * 40, 30), -30))
        
        # Z axis control (up/down)
        # Keep target centered vertically
        z_vel = int(max(min(height_offset * 30, 30), -30))
        
        # Yaw control (rotation)
        # Turn to face target directly
        yaw_vel = int(max(min(heading * 60, 30), -30))
        
        # Send RC control command to Tello
        self.tello.send_rc_control(y_vel, x_vel, z_vel, yaw_vel)
        
        # Update the visualization
        self.visualizer.update_control_values(y_vel, x_vel, z_vel, yaw_vel, time.time())
        
        self.get_logger().debug(f"Tracking: distance={distance:.2f}cm, heading={heading:.2f}, " 
                           f"vels=[{y_vel},{x_vel},{z_vel},{yaw_vel}]")

    def process_frame(self, frame):
        """Process camera frame to detect and track targets"""
        try:
            if frame is None or frame.size == 0:
                self.get_logger().warn("Received empty frame, skipping processing")
                return
                    
            # Log the frame shape
            self.get_logger().debug(f"Processing frame: {frame.shape}")
            
            # Run person detection
            person_tracks = self.person_tracker.process_frame(frame)
            # Add more detailed logging about detections
            if person_tracks:
                self.get_logger().info(f"Detected {len(person_tracks)} persons: {[tr['track_id'] for tr in person_tracks]}")
            else:
                self.get_logger().debug("No persons detected in frame")
            
            # Process detections
            frame_height, frame_width = frame.shape[:2]
            current_time = time.time()
            detected_ids = set()
            
            for track in person_tracks:
                # Extract bounding box and track id
                l, t, r, b = track["bbox"]
                tid = track["track_id"]
                detected_ids.add(tid)
                
                # Calculate center point
                cx = (l + r) / 2
                cy = (t + b) / 2
                
                # Calculate relative position (-1 to 1 coordinates)
                rel_x = (cx - frame_width/2) / (frame_width/2)  # Left-right
                rel_y = (cy - frame_height/2) / (frame_height/2)  # Up-down
                
                # Calculate heading (positive is right, negative is left)
                heading = rel_x
                
                # Calculate distance using bounding box area
                # Second-order polynomial scaling (adjust coefficients based on calibration)
                box_width = r - l
                box_height = b - t
                area = box_width * box_height
                area_ratio = area / (frame_width * frame_height)
                
                # Simple second-order polynomial for distance estimation (in cm)
                # distance = a*(area_ratio)^2 + b*(area_ratio) + c
                # These coefficients should be calibrated for your drone and target size
                a = 25000
                b = -3500
                c = 350
                estimated_distance = a*(area_ratio**2) + b*area_ratio + c
                
                # Clamp the distance to reasonable values (in cm)
                estimated_distance = max(50, min(500, estimated_distance))
                
                # Normalize for our -1 to 1 scale
                rel_z = (estimated_distance - 50) / 450  # Maps 50cm->0.0, 500cm->1.0
                
                # Update target information
                self.targets[tid] = {
                    'position': Point(x=rel_x, y=rel_y, z=rel_z),
                    'heading': heading,
                    'distance': estimated_distance,
                    'timestamp': current_time,
                    'confidence': min(1.0, area_ratio * 20),  # Simple confidence metric
                    'bbox': (l, t, r, b)
                }
                
                self.get_logger().debug(
                    f"Box dimensions: w={r-l}, h={b-t}, area_ratio={area_ratio:.4f}, " 
                    f"estimated_distance={estimated_distance:.2f}cm"
                )
                
                # Handle target assignment and tracking
                self.manage_target_tracking(tid, current_time)
            
            # Handle lost targets
            self.cleanup_lost_targets(detected_ids, current_time)
            
            # Create annotated frame for visualization
            annotated_frame = draw_tracking_status(
                frame, 
                self.current_state.name, 
                self.assigned_target_id,
                self.potential_target_id,
                self.detection_persistence_count,
                self.min_detections_for_tracking,
                self.targets
            )
            
            # Update the visualization
            if annotated_frame is not None:
                self.visualizer.update_frame(annotated_frame)
            
            # Add summary of current tracking state after processing
            self.get_logger().info(f"Current state: {self.current_state.name}, " +
                            f"Assigned target: {self.assigned_target_id}, " +
                            f"Potential target: {self.potential_target_id}, " +
                            f"Persistence: {self.detection_persistence_count}/{self.min_detections_for_tracking}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def state_machine_callback(self):
        """Main state machine execution loop"""
        # Try to get camera frame for states that need it
        if self.current_state in [DroneState.SEARCHING, DroneState.TRACKING]:
            # Make sure camera is initialized
            if not self.camera_initialized:
                if not self.initialize_camera():
                    self.get_logger().warn("Camera not initialized, skipping frame processing")
                    return
                
            # Get frame with appropriate error handling
            try:
                frame = self.tello_camera.get_frame()
                frame_age = self.tello_camera.get_frame_age()
                
                if frame is None:
                    self.get_logger().warn("No frame available")
                elif frame_age > 1.0:
                    self.get_logger().warn(f"Frame too old ({frame_age:.1f}s), skipping")
                else:
                    self.process_frame(frame)
            except Exception as e:
                self.get_logger().error(f"Error getting frame: {str(e)}")
                # If we consistently fail to get frames, try reinitializing
                if not self.camera_error:
                    self.camera_initialized = False
                    self.camera_error = True

        # Execute state-specific behaviors
        if self.current_state == DroneState.IDLE:
            # In IDLE state, wait for initialization command
            pass
            
        elif self.current_state == DroneState.TAKEOFF:
            # Handle takeoff sequence
            try:
                self.get_logger().info("Taking off...")
                self.tello.takeoff()
                self.get_logger().info("Takeoff successful")
                
                # Set initial speed
                self.tello.set_speed(50)
                
                # Initialize camera before searching
                if not self.camera_initialized:
                    if self.initialize_camera():
                        self.get_logger().info("Camera initialized successfully")
                    else:
                        self.get_logger().warn("Camera initialization failed, continuing anyway")
                
                # Move to searching state
                self.change_state(DroneState.SEARCHING)
            except Exception as e:
                self.get_logger().error(f"Takeoff failed: {str(e)}")
                self.change_state(DroneState.IDLE)
            
        elif self.current_state == DroneState.SEARCHING:
            # Execute search pattern even if frame processing fails
            self.execute_search_pattern()
            
            # If we have an assignment, transition to tracking
            if self.assigned_target_id is not None:
                self.change_state(DroneState.TRACKING)
        
        elif self.current_state == DroneState.WAITING_FOR_ASSIGNMENT:
            # Wait for target assignment from prioritization module
            # Hover in place
            self.tello.send_rc_control(0, 0, 0, 0)
            
            # Update visualization
            self.visualizer.update_control_values(0, 0, 0, 0, time.time())
            
            # If targets are detected but none assigned, request assignment
            if self.targets and self.assigned_target_id is None:
                self.get_logger().debug("Waiting for target assignment...")
            
            # If assignment received, transition to tracking
            if self.assigned_target_id is not None:
                self.change_state(DroneState.TRACKING)
        
        elif self.current_state == DroneState.TRACKING:
            
            # Check if target is still visible
            if self.check_target_timeout():
                self.get_logger().warn("Target lost, returning to search")
                self.change_state(DroneState.SEARCHING)
                return
                    
            # Execute tracking behavior if target exists
            if self.assigned_target_id in self.targets:
                self.execute_tracking()
            else:
                # Hover in place if target temporarily not visible
                self.tello.send_rc_control(0, 0, 0, 0)
                
                # Update visualization
                self.visualizer.update_control_values(0, 0, 0, 0, time.time())
        
        elif self.current_state == DroneState.LANDING:
            # Execute landing procedure
            try:
                self.tello.land()
                self.get_logger().info("Landing successful")
                self.change_state(DroneState.IDLE)
            except Exception as e:
                self.get_logger().error(f"Landing failed: {str(e)}")
        
        elif self.current_state == DroneState.EMERGENCY:
            # Execute emergency landing
            try:
                self.tello.emergency()
                self.get_logger().warn("Emergency landing executed")
            except Exception as e:
                self.get_logger().error(f"Emergency command failed: {str(e)}")
            finally:
                self.change_state(DroneState.IDLE)
    
    def manage_target_tracking(self, tid, current_time):
        """Manage target assignment and tracking persistence"""
        # If already tracking this target, update last seen time
        if self.assigned_target_id == tid:
            self.target_last_seen = current_time
            self.detection_persistence_count = self.min_detections_for_tracking  # Keep at max
            self.get_logger().debug(f"Already tracking target {tid}, updated last_seen time")
            return
        
        # If searching and no target assigned yet, consider this one
        if self.current_state == DroneState.SEARCHING and self.assigned_target_id is None:
            # If this is the first detection or we're seeing a new target
            if self.potential_target_id is None:
                # Start tracking a new potential target
                self.potential_target_id = tid
                self.detection_persistence_count = 1
                self.get_logger().info(f"New potential target ID {tid}, detection count: 1")
            # If we're seeing a different track ID but it's likely the same person (YOLO DeepSORT reidentification issue)
            # This is the key change - we'll continue persistence even with different track IDs
            else:
                # Check if the new detection is close to the previous potential target
                if tid in self.targets and self.potential_target_id in self.targets:
                    # Get positions
                    new_pos = self.targets[tid]['position']
                    old_pos = self.targets[self.potential_target_id]['position']
                    
                    # Calculate position difference
                    dist_x = abs(new_pos.x - old_pos.x)
                    dist_y = abs(new_pos.y - old_pos.y)
                    
                    # If within reasonable distance, consider it the same person
                    if dist_x < 0.4 and dist_y < 0.4:  # Threshold can be adjusted
                        self.get_logger().info(f"Track ID changed from {self.potential_target_id} to {tid} but appears to be same person")
                        self.potential_target_id = tid  # Update to new ID
                        self.detection_persistence_count += 1
                        self.get_logger().info(f"Continuing with potential target ID {tid}, detection count: {self.detection_persistence_count}")
                    else:
                        # New person, restart persistence
                        self.potential_target_id = tid
                        self.detection_persistence_count = 1
                        self.get_logger().info(f"Detected new person (ID {tid}), restarting persistence count: 1")
                else:
                    # Simple case - just update the ID and increment
                    self.potential_target_id = tid
                    self.detection_persistence_count += 1
                    self.get_logger().info(f"Updated potential target to ID {tid}, detection count: {self.detection_persistence_count}")
                
            # Check if we've met the persistence threshold
            if self.detection_persistence_count >= self.min_detections_for_tracking:
                self.assigned_target_id = tid
                self.target_last_seen = current_time
                self.get_logger().info(f"Target {tid} confirmed after {self.detection_persistence_count} detections, transitioning to TRACKING")
                # Force state change to tracking
                self.change_state(DroneState.TRACKING)

    def cleanup_lost_targets(self, detected_ids, current_time):
        """Clean up lost targets and handle persistence, with improved logic for track ID changes"""
        # Don't immediately reset potential target if it disappears momentarily
        # Instead, give it a grace period to reappear with possibly a different ID
        if self.potential_target_id is not None and self.potential_target_id not in detected_ids:
            # Check if we have this attribute, if not create it
            if not hasattr(self, 'potential_target_last_seen'):
                self.potential_target_last_seen = current_time
                
            # Only reset if the potential target has been missing for more than 1 second
            if current_time - self.potential_target_last_seen > 1.0:
                self.get_logger().debug(f"Potential target {self.potential_target_id} lost for >1s, resetting persistence")
                self.potential_target_id = None
                self.detection_persistence_count = 0
        else:
            # Update the last seen time for the potential target
            if self.potential_target_id is not None:
                self.potential_target_last_seen = current_time
        
        # Remove old targets from dictionary (except current tracking target)
        for tid in list(self.targets.keys()):
            if tid not in detected_ids and current_time - self.targets[tid]['timestamp'] > 2.0:
                if tid != self.assigned_target_id:  # Keep assigned target longer
                    del self.targets[tid]

    def start_mission(self):
        if self.current_state == DroneState.IDLE:
            self.get_logger().info("Starting mission")
            
            # Set a lower detection persistence threshold to make it easier to track targets
            self.min_detections_for_tracking = 3 
            self.get_logger().info(f"Setting min_detections_for_tracking to {self.min_detections_for_tracking}")
            
            # Initialize camera here for early detection of issues
            if self.initialize_camera():
                self.get_logger().info("Camera initialized successfully")
            else:
                self.get_logger().warn("Camera initialization failed, continuing anyway")
                
            self.change_state(DroneState.TAKEOFF)
    
    def abort_mission(self):
        """Abort the drone mission and land"""
        self.get_logger().info("Aborting mission")
        self.change_state(DroneState.LANDING)
    
    def cleanup(self):
        """Clean up resources and land the drone"""
        # Stop visualizer
        if hasattr(self, 'visualizer'):
            self.visualizer.stop()
        
        # Stop camera stream
        if hasattr(self, 'tello_camera') and self.camera_initialized:
            try:
                self.tello_camera.stop()
                self.camera_initialized = False
            except Exception as e:
                self.get_logger().error(f"Error stopping camera: {str(e)}")
        
        # Land the drone if flying
        if self.current_state not in [DroneState.IDLE, DroneState.LANDING]:
            try:
                self.tello.land()
            except Exception:
                self.get_logger().error("Failed to land drone during cleanup")
        
        # End the Tello connection
        try:
            self.tello.end()
        except Exception:
            self.get_logger().error("Failed to properly end Tello connection")

    def get_position(self):
        """Get the current position of the drone"""
        try:
            if all(v is not None for v in self.velocity.values()):
                self.position["x"] += self.velocity["x"] * 0.1
                self.position["y"] += self.velocity["y"] * 0.1
                self.position["z"] = self.tello.get_height()
                self.logger().info(f"Current position: {self.position}")
        except Exception as e:
            self.get_logger().error(f"Error getting position: {str(e)}")
            return None

def main(args=None):
    rclpy.init(args=args)
    tello_state_machine = TelloStateMachine()
    
    try:
        # Run the state machine
        rclpy.spin(tello_state_machine)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup on shutdown
        tello_state_machine.cleanup()
        tello_state_machine.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()