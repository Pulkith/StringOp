from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tello_target_tracker',
            executable='tello_state_machine',
            name='tello_state_machine',
            parameters=[
                {'drone_id': 1},
                {'net_interface': 'wlxdc6279da55db'}
            ],
            output='screen'
        ),
    ])