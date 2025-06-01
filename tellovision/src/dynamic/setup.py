from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dynamic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # To install launch files:
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # To install models into the package's share directory:
        (os.path.join('share', package_name, 'models'), [
            'models/dynamic_gesture_model_best_loss.pth',
            'models/normalization_params.npz'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zheng',
    maintainer_email='zooi6811@uni.sydney.edu.au',
    description='TODO: Package description',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamic_node = dynamic.dynamic_node:main'
        ],
    },
)
