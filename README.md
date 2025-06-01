# TelloVision

TelloVision is a ROS2-based project designed to control a DJI Tello drone using advanced computer vision techniques. It enables real-time drone control through both static and dynamic hand gestures, as well as autonomous face tracking, all orchestrated within the ROS2 framework.

## Features

* **Gesture-Based Control**: The drone can be piloted using a variety of hand gestures.
    * **Static Gesture Recognition**: A neural network model recognises static hand poses (e.g., 'ok', 'peace', 'fist') to switch between control modes or issue specific commands.
    * **Dynamic Gesture Recognition**: An LSTM-based model detects dynamic movements like swipes and waves to execute flight manoeuvres.
    * **Directional Gestures**: Simple pointing gestures are used for directional commands (up, down, left, right).

* **Autonomous Face Tracking**: In this mode, the drone can detect a face and automatically adjust its position to keep the person in the frame, using a PID controller for smooth movement.

* **State-Driven Operation**: A robust state machine manages the drone's operational mode, seamlessly switching between:
    * `IDLE`: The drone hovers and awaits a command.
    * `GESTURE_MOVEMENT`: The drone responds to directional hand gestures.
    * `DYNAMIC_GESTURE`: The drone responds to dynamic hand movements.
    * `FACE_TRACK`: The drone autonomously follows a detected face.

* **Manual Override**: A keyboard-based control node is available for direct manual piloting.

## Dependencies

### System Dependencies
* **ROS2 Humble**: The project is built and tested on ROS2 Humble.
* **Python**: Python 3.8 or newer.
* **OpenCV**: For real-time computer vision tasks.

## Usage

The primary way to run the project is by using the provided launch file, which starts all the necessary nodes for drone control and gesture recognition.

1.  **Connect to the Tello Drone's Wi-Fi**:
    Ensure your computer is connected to the Tello drone's Wi-Fi network.

2.  **Launch the System**:
    Execute the main launch file from the `orchestrator_launch` package:
    ```bash
    ros2 launch orchestrator_launch bringup.launch.py
    ```

3.  **Controlling the Drone**:
    * **Takeoff**: Use the manual control node (details may need to be added on how to trigger this) or a pre-defined gesture to take off.
    * **Switching States**: Use static hand gestures to change the drone's mode:
        * **Peace Sign (`peace`)**: Return the drone to the `IDLE` state.
        * **OK Sign (`ok`)**: Activate `FACE_TRACK` mode.
        * **Fingers Crossed (`fingers_crossed`)**: Activate `GESTURE_MOVEMENT` mode.
        * **Animal Sign (`animal`)**: Activate `DYNAMIC_GESTURE` mode.
    * **Landing**: The `Spider` gesture or a `wave` in dynamic mode will land the drone safely.
