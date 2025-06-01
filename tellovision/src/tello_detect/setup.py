from setuptools import setup
import os
from glob import glob

package_name = 'tello_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add model files under share/<package_name>/models
        (os.path.join('share', package_name, 'models'), glob(os.path.join(package_name, 'models', '*'))),
        # Add launch files under share/<package_name>/launch
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=[
        'setuptools', 'opencv-python', 'mediapipe', 'numpy', 'torch', 'joblib', 'ament_index_python'
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Package for controlling a Tello drone using gesture detection.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'movement_gestures = tello_detect.movement_gestures:main',
            'simulator = tello_detect.sim:main',
            'handface = tello_detect.handface_filter:main',
            'simple_predict = tello_detect.simple_predict:main',
            'state_control = tello_detect.state_control:main',
            'face_rotate = tello_detect.face_rotate:main',
        ],
    },
)

