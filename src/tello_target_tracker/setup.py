from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tello_target_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=['tello_target_tracker', 'tello_target_tracker.DJITelloPy.djitellopy'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),  # Changed to use proper marker file
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A ROS2 package for Tello drones to track and follow targets',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tello_state_machine = tello_target_tracker.tello_state_machine:main',
        ],
    },
)