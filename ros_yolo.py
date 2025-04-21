#!/usr/bin/env python3

import numpy as np
import cv2
from ultralytics import YOLO
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped


class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__("object_recognition_node")
        self.declare_parameter("target_object", "person")

        self.get_logger().info("Object recognition node has been started.")

        # Create a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the RGB and depth image topics (synchronized via message_filters)
        self.rgb_sub = self.create_subscription(
            Image, "/image_raw", self.immage_callback, 10
        )
        self.ts.registerCallback(self.image_callback)

        # Publisher for detected objects
        self.detected_objects_pub = self.create_publisher(
            String, "detected_objects", 10
        )

        # Set up TF2 for transforming points
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load YOLO v8 model
        yolo_model_path = "yolov8n.pt"
        self.yolo_model = YOLO(yolo_model_path)
        self.get_logger().info("YOLO v8 model loaded.")

        # Confidence threshold for YOLO detection
        self.confidence_threshold = 0.5

        # Target object
        self.target_object = self.get_parameter("target_object").value

        self.get_logger().info(f"Looking for target object: {self.target_object}")

    def image_callback(self, rgb_msg):
        self.get_logger().info("Received synchronized RGB and depth data.")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error("RGB image conversion failed: " + str(e))
            return

        # Run YOLO detection on the full image
        results = self.yolo_model(cv_image, verbose=False)
        
        detected_objects = []

        # Process YOLO detections from the full image
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Iterate over each detected object
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy().item())
                label = self.yolo_model.names[class_id]
                confidence = box.conf[0].cpu().numpy().item()

                self.get_logger().info(
                    f"Detected {label} with confidence {confidence:.2f} in full image."
                )

                if (
                    self.target_object in label.lower()
                    and confidence > self.confidence_threshold
                ):
                    # Get the bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    center_u = int((x1 + x2) / 2)
                    center_v = int((y1 + y2) / 2)

                    # TODO

                    self.get_logger().info(
                        "Object detected: " # TODO
                    )
                    stamp = self.get_clock().now().to_msg()
                    detected_objects.append(
                        {
                            "detection_time": stamp.sec + stamp.nanosec / 1e9,
                            "object": self.target_object,
                            "position/bbox/whatever we want to publish": {
                                # TODO
                            },
                        }
                    )

        # Publish detected objects if any were found
        if detected_objects:
            msg = String()
            msg.data = json.dumps(detected_objects)
            self.detected_objects_pub.publish(msg)
            self.get_logger().info(f"Detected objects: {detected_objects}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()






