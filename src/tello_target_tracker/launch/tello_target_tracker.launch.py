from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'drone_id',
            default_value='0',
            description='Drone ID for multi-drone setups'
        ),
        
        # Tello state machine node
        Node(
            package='tello_target_tracker',
            executable='tello_state_machine',
            name='tello_state_machine',
            parameters=[
                {'drone_id': LaunchConfiguration('drone_id')}
            ],
            output='screen'
        ),
    ])