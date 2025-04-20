import numpy as np
import cv2
from collections import deque
import time

# For ROS Image publishing
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

class RosVisualizer:
    """
    Class to handle visualization for the Tello drone tracking system using ROS topics
    instead of OpenCV windows. This avoids Qt/threading issues.
    """
    def __init__(self, node, history_length=100):
        # Store the ROS node
        self.node = node
        
        # Control command history
        self.history_length = history_length
        self.x_history = deque(maxlen=history_length)  # Forward/backward
        self.y_history = deque(maxlen=history_length)  # Left/right
        self.z_history = deque(maxlen=history_length)  # Up/down
        self.yaw_history = deque(maxlen=history_length)  # Rotation
        self.time_history = deque(maxlen=history_length)
        
        # Last update times
        self.last_control_update = 0
        
        # CV Bridge for converting images
        self.bridge = CvBridge()
        
        # Create publishers
        self.frame_pub = node.create_publisher(
            Image, 
            f'/drone_{node.drone_id}/annotated_frame', 
            10
        )
        self.control_pub = node.create_publisher(
            Float32MultiArray,
            f'/drone_{node.drone_id}/control_values',
            10
        )
        
        # Flag for running
        self.running = True
        
        node.get_logger().info("ROS Visualizer initialized")
            
    def start(self):
        """Start the visualization"""
        self.running = True
            
    def stop(self):
        """Stop visualization"""
        self.running = False
    
    def update_frame(self, frame):
        """Publish the frame to a ROS topic"""
        if frame is None or not self.running:
            return
            
        try:
            # Convert frame to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            
            # Publish the image
            self.frame_pub.publish(ros_image)
        except Exception as e:
            self.node.get_logger().error(f"Error publishing frame: {str(e)}")
    
    def update_control_values(self, y_vel, x_vel, z_vel, yaw_vel, timestamp):
        """Update the control values history and publish to ROS"""
        if not self.running:
            return
            
        # Update history
        self.x_history.append(x_vel)
        self.y_history.append(y_vel)
        self.z_history.append(z_vel)
        self.yaw_history.append(yaw_vel)
        self.time_history.append(timestamp)
        
        # Only publish occasionally to avoid flooding the network
        current_time = time.time()
        if current_time - self.last_control_update > 0.1:  # 10Hz
            try:
                # Create control values message
                control_msg = Float32MultiArray()
                
                # Set up dimensions
                control_msg.layout.dim = [
                    MultiArrayDimension(
                        label="control_values",
                        size=4,
                        stride=4
                    )
                ]
                
                # Add the current control values
                control_msg.data = [float(x_vel), float(y_vel), float(z_vel), float(yaw_vel)]
                
                # Publish
                self.control_pub.publish(control_msg)
                self.last_control_update = current_time
            except Exception as e:
                self.node.get_logger().error(f"Error publishing control values: {str(e)}")


def draw_tracking_status(frame, state, assigned_target_id, potential_target_id, 
                         detection_count, min_detections, targets):
    """Draw tracking status information on the frame for debugging"""
    if frame is None:
        return None
        
    # Create a copy to avoid modifying the original
    vis_frame = frame.copy()
    
    # Draw state information
    cv2.putText(
        vis_frame,
        f"State: {state}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    
    # Draw target information
    target_text = f"Target: {assigned_target_id if assigned_target_id is not None else 'None'}"
    cv2.putText(
        vis_frame,
        target_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    
    # Draw potential target and persistence
    cv2.putText(
        vis_frame,
        f"Potential: {potential_target_id}, Count: {detection_count}/{min_detections}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    
    # Draw detected targets
    for tid, target_info in targets.items():
        if 'bbox' in target_info:
            l, t, r, b = target_info['bbox']
            
            # Use different colors for assigned, potential, and other targets
            if assigned_target_id == tid:
                color = (0, 255, 0)  # Green for assigned target
            elif potential_target_id == tid:
                color = (0, 255, 255)  # Yellow for potential target
            else:
                color = (255, 0, 0)  # Blue for other targets
                
            # Draw bounding box
            cv2.rectangle(vis_frame, (l, t), (r, b), color, 2)
            
            # Display ID and distance
            if 'distance' in target_info:
                distance = target_info['distance']
                cv2.putText(
                    vis_frame,
                    f"ID:{tid} ({distance:.0f}cm)",
                    (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            else:
                cv2.putText(
                    vis_frame,
                    f"ID:{tid}",
                    (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Draw target center
            center_x = (l + r) // 2
            center_y = (t + b) // 2
            cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
            
            # Draw frame center
            h, w = vis_frame.shape[:2]
            cv2.circle(vis_frame, (w//2, h//2), 5, (0, 0, 255), -1)
            cv2.line(vis_frame, (w//2, h//2), (center_x, center_y), (255, 0, 255), 1)
    
    return vis_frame