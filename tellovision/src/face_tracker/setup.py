from setuptools import setup

package_name = 'face_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'mediapipe',
        'djitellopy',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Young',
    maintainer_email='younghu27@gmail.com',
    description='ROS2 face‐tracking control for Tello using MediaPipe',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'face_tracker = face_tracker.face_tracker:main',
        ],
    },
)
