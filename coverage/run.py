import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import pose_graph

class DroneCoordinator(Node):
    def __init__(self):
        super().__init__('drone_coordinator')
        self.state = pose_graph.GlobalState()

        # Subscribers and Publishers
        self.drone_state_subs = []
        self.drone_action_pubs = []

        for i in range(3):
            drone_id = f"drone{i}"
            # Subscribers
            self.drone_state_subs.append(
            self.create_subscription(
                String,
                f'{drone_id}_state',
                lambda msg, drone_id=drone_id: self.drone_state_callback(msg, drone_id),
                10
            )
            )
            # Publishers
            self.drone_action_pubs.append(
                self.create_publisher(String, f'{drone_id}_action', 10)
            )

        self.get_logger().info("Drone Coordinator Node has been started.")
        
    def drone_state_callback(self, msg, drone_id):
        self.get_logger().info(f"Received {drone_id} state: {msg.data}")
        drone_id = msg.data.split()[0]

        #TODO: process msg
        self.state.update_drone_pose(drone_id, msg.data)
        self.state.update_drone_objects(drone_id, msg.data)
        action = self.state.get_action(drone_id)

        self.get_logger().info(f"Publishing action for {drone_id}: {action}")
        action_msg = String()
        action_msg.data = action
        self.drone_action_pubs[drone_id].publish(action_msg)

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