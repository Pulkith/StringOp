from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tello_target_tracker',
            executable='tello_state_machine',
            name='tello_state_machine',
            parameters=[
                {'drone_id': 0},
                {'net_interface': 'wlp4s0'}
            ],
            output='screen'
        ),
    ])