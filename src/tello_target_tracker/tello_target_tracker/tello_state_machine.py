#!/usr/bin/env python3

import enum
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, Twist, Vector3
from tello_interfaces.msg import TargetInfo, DroneStatus

# Add this import
from std_srvs.srv import SetBool


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
        
        # Initialize Tello drone
        self.tello = Tello()
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
        self.target_detect_sub = self.create_subscription(
            TargetInfo,
            f'/perception/targets',
            self.target_detection_callback,
            10
        )
        self.target_assign_sub = self.create_subscription(
            String,
            f'/target_prioritization/assignments',
            self.target_assignment_callback,
            10
        )
        self.emergency_sub = self.create_subscription(
            Bool,
            '/emergency',
            self.emergency_callback,
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
    
    def target_detection_callback(self, msg):
        """Process target detection from perception module"""
        # Update the targets dictionary with new detection
        self.targets[msg.target_id] = {
            'position': msg.relative_position,
            'timestamp': time.time(),
            'confidence': msg.confidence
        }
        
        # If tracking this target, update last seen time
        if self.assigned_target_id is not None and msg.target_id == self.assigned_target_id:
            self.target_last_seen = time.time()
            self.get_logger().debug(f"Tracking target {self.assigned_target_id} at position: "
                                  f"x={msg.relative_position.x}, y={msg.relative_position.y}, z={msg.relative_position.z}")
    
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
    
    def emergency_callback(self, msg):
        """Handle emergency signals"""
        if msg.data:
            self.get_logger().warn("Emergency signal received!")
            self.change_state(DroneState.EMERGENCY)
    
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
    
    def check_target_timeout(self):
        """Check if the tracked target has timed out"""
        if self.assigned_target_id is not None:
            # If target hasn't been seen for the threshold time
            if time.time() - self.target_last_seen > self.target_lost_threshold:
                self.get_logger().warn(f"Target {self.assigned_target_id} lost! Returning to search.")
                self.assigned_target_id = None
                return True
        return False
    
    def execute_search_pattern(self):
        """Execute a search pattern to find targets"""
        # Use Tello's RC control for rotation search
        # Rotate in place to look for targets
        self.tello.send_rc_control(0, 0, 0, 30 * self.search_direction)
        
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
        """Execute tracking behavior based on target position"""
        if self.assigned_target_id is None or self.assigned_target_id not in self.targets:
            return
            
        target = self.targets[self.assigned_target_id]
        pos = target['position']
        
        # Braitenberg-inspired following behavior
        # Convert position to velocity commands
        
        # X axis control (forward/backward)
        # Maintain distance from target
        target_distance = 100.0  # Desired distance in cm
        x_vel = int(max(min((pos.z - target_distance) * 0.3, 30), -30))
        
        # Y axis control (left/right)
        y_vel = int(max(min(-pos.x * 0.3, 30), -30))
        
        # Z axis control (up/down)
        z_vel = int(max(min(pos.y * 0.3, 30), -30))
        
        # Yaw control to keep facing target
        yaw_vel = int(max(min(pos.x * 0.5, 30), -30))
        
        # Send RC control command to Tello
        self.tello.send_rc_control(y_vel, x_vel, z_vel, yaw_vel)
        
        # Also publish for visualization/debugging
        cmd = Twist()
        cmd.linear.x = x_vel / 100.0  # Normalize to -1.0 to 1.0 for ROS
        cmd.linear.y = y_vel / 100.0
        cmd.linear.z = z_vel / 100.0
        cmd.angular.z = yaw_vel / 100.0
        self.cmd_vel_pub.publish(cmd)
    
    def state_machine_callback(self):
        """Main state machine execution loop"""
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
                
                # Move to searching state
                self.change_state(DroneState.SEARCHING)
            except Exception as e:
                self.get_logger().error(f"Takeoff failed: {str(e)}")
                self.change_state(DroneState.IDLE)
            
        elif self.current_state == DroneState.SEARCHING:

            # Check if any targets are detected (without waiting for assignment)
            if self.targets:
                # Simply take the first detected target
                self.assigned_target_id = list(self.targets.keys())[0]
                self.get_logger().info(f"Target {self.assigned_target_id} detected, tracking immediately")
                self.change_state(DroneState.TRACKING)
            else:
                # Continue searching if no targets found
                self.execute_search_pattern()


            # Search for targets if none assigned
            # if self.assigned_target_id is None:
            #     self.execute_search_pattern()
            # else:
            #     # If we have an assignment, transition to tracking
            #     self.change_state(DroneState.TRACKING)
        
        elif self.current_state == DroneState.WAITING_FOR_ASSIGNMENT:
            # Wait for target assignment from prioritization module
            # Hover in place
            self.tello.send_rc_control(0, 0, 0, 0)
            
            # If targets are detected but none assigned, request assignment
            if self.targets and self.assigned_target_id is None:
                self.get_logger().debug("Waiting for target assignment...")
            
            # If assignment received, transition to tracking
            if self.assigned_target_id is not None:
                self.change_state(DroneState.TRACKING)
        
        elif self.current_state == DroneState.TRACKING:
            # Check if target is still visible
            if self.check_target_timeout():
                self.change_state(DroneState.SEARCHING)
                return
                
            # Execute tracking behavior
            self.execute_tracking()
        
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
    
    def start_mission(self):
        """Start the drone mission"""
        if self.current_state == DroneState.IDLE:
            self.get_logger().info("Starting mission")
            self.change_state(DroneState.TAKEOFF)
    
    def abort_mission(self):
        """Abort the drone mission and land"""
        self.get_logger().info("Aborting mission")
        self.change_state(DroneState.LANDING)
    
    def cleanup(self):
        """Clean up resources and land the drone"""
        if self.current_state not in [DroneState.IDLE, DroneState.LANDING]:
            try:
                self.tello.land()
            except Exception:
                self.get_logger().error("Failed to land drone during cleanup")
        
        try:
            self.tello.end()
        except Exception:
            self.get_logger().error("Failed to properly end Tello connection")

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