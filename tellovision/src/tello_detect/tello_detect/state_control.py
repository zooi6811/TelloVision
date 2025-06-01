import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from enum import Enum


class State(Enum):
    IDLE = 0
    GESTURE_MOVEMENT = 1
    DYNAMIC_GESTURE = 2
    FACE_TRACK = 3


class GestureStateNode(Node):

    def __init__(self):
        super().__init__('gesture_state_node')

        # Current state (start in Idle)
        self.state = State.IDLE

        # Subscribers
        self.gesture_simple_sub = self.create_subscription(
            String,
            '/control_gesture',
            self.gesture_simple_callback,
            10
        )

        self.gesture_sub = self.create_subscription(
            String,
            '/movement_gesture',
            self.gesture_callback,
            10
        )

        self.dynamic_gesture_sub = self.create_subscription(
            String,
            '/dynamic_gesture_input',
            self.dynamic_gesture_callback,
            10
        )

        # Publishers
        self.movement_command_pub = self.create_publisher(
            String,
            '/bob',
            10
        )

        self.face_track_control_pub = self.create_publisher(
            String,
            '/face_enable',
            10
        )

        # Timer for face tracking control status
        self.create_timer(0.5, self.face_track_callback)

        self.get_logger().info('Gesture State Node with Enum has started.')

    def gesture_simple_callback(self, msg: String):
        gesture = msg.data
        self.get_logger().info(f'[gesture_simple] Received: {gesture} | Current state: {self.state.name}')

        if gesture == 'peace':
            self.state = State.IDLE
            self.get_logger().info('Transitioning to IDLE')
            return

        if self.state == State.IDLE:
            if gesture == 'fingers_crossed':
                self.state = State.GESTURE_MOVEMENT
                self.get_logger().info('Transitioning to GESTURE_MOVEMENT')
            elif gesture == 'animal':
                self.state = State.DYNAMIC_GESTURE
                self.get_logger().info('Transitioning to DYNAMIC_GESTURE')
            elif gesture == 'ok':
                self.state = State.FACE_TRACK
                self.get_logger().info('Transitioning to FACE_TRACK')

    def gesture_callback(self, msg: String):
        if self.state != State.GESTURE_MOVEMENT:
            return

        gesture = msg.data
        self.get_logger().info(f'[gesture] Received in GESTURE_MOVEMENT: {gesture}')
        command_msg = String()

        if gesture == 'Spider':
            command_msg.data = 'land'
        elif gesture == 'Peace':
            command_msg.data = 'move_forward'
        elif gesture == 'Kawaii':
            command_msg.data = 'move_back'
        elif gesture == 'Assert_Dominance':
            command_msg.data = 'flip_back'
        elif gesture == 'Point_Left' or gesture == 'Thumbs_Left':
            command_msg.data = 'move_left'
        elif gesture == 'Point_Right' or gesture == 'Thumbs_Right':
            command_msg.data = 'move_right'
        elif gesture == 'Point_Up' or gesture == 'Thumbs_Up':
            command_msg.data = 'move_up'
        elif gesture == 'Point_Down' or gesture == 'Thumbs_Down':
            command_msg.data = 'move_down'
        else:
            self.get_logger().info(f'Unknown movement gesture: {gesture}')
            return

        self.movement_command_pub.publish(command_msg)
        self.get_logger().info(f'Published movement_command: {command_msg.data}')

    def dynamic_gesture_callback(self, msg: String):
        if self.state != State.DYNAMIC_GESTURE:
            return

        gesture = msg.data
        self.get_logger().info(f'[dynamic_gesture] Received in DYNAMIC_GESTURE: {gesture}')
        command_msg = String()

        if gesture == 'circle':
            command_msg.data = 'move_back'
        elif gesture == 'swipe_side':
            command_msg.data = 'move_left'
        elif gesture == 'wave':
            command_msg.data = 'land'
        elif gesture == 'swipe_vert':
            command_msg.data = 'flip_back'
        else:
            self.get_logger().info(f'Unknown dynamic gesture: {gesture}')
            return

        self.movement_command_pub.publish(command_msg)
        self.get_logger().info(f'Published movement_command: {command_msg.data}')


    def face_track_callback(self):
        """Publishes 'face_tracking' when active, otherwise 'invalid'."""
        msg = String()
        msg.data = 'face_tracking' if self.state == State.FACE_TRACK else 'invalid'
        self.face_track_control_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

