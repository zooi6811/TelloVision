from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tello_detect',
            executable='movement_gestures',
            name='movement_gestures_node',
            output='screen'
        ),
        Node(
            package='tello_detect',
            executable='simulator',
            name='simulator_node',
            output='screen'
        ),
        Node(
            package='tello_detect',
            executable='handface',
            name='handface_node',
            output='screen'
        ),
        Node(
            package='tello_detect',
            executable='simple_predict',
            name='simple_predict_node',
            output='screen'
        ),
        Node(
            package='tello_detect',
            executable='state_control',
            name='state_control_node',
            output='screen'
        ),
    ])

