import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tello_interfaces.msg import DroneStatus, DroneAction
import numpy as np
import pose_graph
from convergence_visualization.simulations import weights_sim

class DroneCoordinator(Node):
    def __init__(self):
        super().__init__('drone_coordinator')
        self.state = pose_graph.GlobalState()

        # Subscriber
        self.drone_state_sub = self.create_subscription(
            DroneStatus,
            f'drone_status',
            lambda msg: self.drone_state_callback(msg),
            10
        )
        # Publisher
        self.drone_action_pub = self.create_publisher(DroneAction, f'action', 10)

        self.get_logger().info("Drone Coordinator Node has been started.")

        self.drones = []
        self.counter = 0
        
    def drone_state_callback(self, msg):
        # Process the drone state message
        data = msg.data
        drone_id = data.drone_id
        
        self.get_logger().info(f"Received {drone_id} state: {msg.data}")
      
        self.state.update_drone_pose(drone_id, pose_graph.GlobalPose(
            x=data.pose.x,
            y=data.pose.y,
            z=data.pose.z,
            yaw=data.pose.yaw
        ))

        objects = []

        for obj in data.objects:
            angle = obj.angle
            distance = obj.distance

            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            object_pose = pose_graph.LocalPose(
                x=x,
                y=y,
                z=0,
                yaw=0
            )
            object_attributes = pose_graph.ObjectAttributes(
                r=obj.r,
                g=obj.g,
                b=obj.b,
            )
            objects.append((object_pose, object_attributes))

        self.state.update_drone_objects(drone_id, objects)
        self.counter += 1

        if self.counter % 5 == 0:
            goals = weights_sim(
                drone_positions=[
                    (self.state.drones[drone_id].x, self.state.drones[drone_id].y)
                    for drone_id in self.state.drones.keys()
                ],
                poi_positions=[
                    (obj[0].x, obj[0].y)
                    for obj in self.state.objects
                ]
            )

            for index, goal in enumerate(goals):
                msg = DroneAction()
                msg.drone_id = drone_id
                msg.object_id = goal

                self.drone_action_pub.publish(msg)
                self.get_logger().info(f"Updated state for {drone_id} with {len(objects)} objects.")


def main(args=None):
    rclpy.init(args=args)
    node = DroneCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Drone Coordinator Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()