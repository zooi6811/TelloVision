from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch the dynamic node
    dynamic_node = Node(
        package='dynamic',
        executable='dynamic_node', 
        name='dynamic_node',
        output='screen'
    )

    # Launch the launch file from tello_detect
    tello_detect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tello_detect'),
                'launch',
                'launch.py'  
            )
        )
    )

    return LaunchDescription([
        dynamic_node,
        tello_detect_launch,
    ])

