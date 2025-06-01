#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp
from std_msgs.msg import String

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from media_pipe_ros2_msg.msg import HandPoint, MediaPipeHumanHand, MediaPipeHumanHandList

class HandsPublisher(Node):
    def __init__(self):
        super().__init__('mediapipe_publisher')

        # Publisher for detected hand landmarks
        self.landmark_pub = self.create_publisher(
            MediaPipeHumanHandList,
            '/mediapipe/human_hand_list',
            10
        )

        # Publisher for detected gesture
        self.gesture_pub = self.create_publisher(
            String,
            '/hand_gesture',
            10
        )

        # Bridge to convert ROS Image to OpenCV
        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.sub_img = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # MediaPipe Hands setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )

    def detect_gesture(self, landmarks_list):
        # """
        # Analyze MediaPipe landmarks and return a gesture label:
        # - 'no_hand'
        # - 'fist'
        # - 'open_palm'
        # - '<finger>_up' for single finger gestures
        # - 'multiple_up' if multiple but not all
        # """
        if not landmarks_list:
            return "no_hand"
        lm = landmarks_list[0].landmark

        # Define finger tip and joint pairs for detection
        fingers = {
            'thumb': (self.mp_hands.HandLandmark.THUMB_TIP, self.mp_hands.HandLandmark.THUMB_IP),
            'index': (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            'middle': (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            'ring': (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            'pinky': (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_DIP)
        }
        up_fingers = []
        # Check each finger for extension
        for name, (tip, joint) in fingers.items():
            if name == 'thumb':
                # Thumb: check horizontal extension for flipped image
                if lm[tip].x < lm[joint].x:
                    up_fingers.append('thumb')
            else:
                # Other fingers: tip above joint in image space
                if lm[tip].y < lm[joint].y:
                    up_fingers.append(name)

        # Single-finger gestures
        if up_fingers == ['thumb']:
            return 'thumb_up'
        if up_fingers == ['index']:
            return 'index_up'
        if up_fingers == ['middle']:
            return 'middle_up'
        if up_fingers == ['ring']:
            return 'ring_up'
        if up_fingers == ['pinky']:
            return 'pinky_up'

        # All or none
        if len(up_fingers) == 0:
            return 'fist'
        if len(up_fingers) == 5:
            return 'open_palm'

        # Multiple but not all
        return 'multiple_up'

    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV BGR frame
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Preprocess for MediaPipe
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        # Convert back to BGR to display
        display_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Prepare landmark message
        human_list = MediaPipeHumanHandList()
        human = MediaPipeHumanHand()
        hand_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):
                label = handedness.classification[0].label
                self.mp_drawing.draw_landmarks(
                    display_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                for idx, lm in enumerate(hand_landmarks.landmark):
                    if label == 'Right':
                        human.right_hand_key_points[idx].name = str(idx)
                        human.right_hand_key_points[idx].x = lm.x
                        human.right_hand_key_points[idx].y = lm.y
                        human.right_hand_key_points[idx].z = lm.z
                        if hand_count == 0:
                            human.left_hand_key_points[idx].name = str(idx)
                            human.left_hand_key_points[idx].x = 0.0
                            human.left_hand_key_points[idx].y = 0.0
                            human.left_hand_key_points[idx].z = 0.0
                    else:
                        human.left_hand_key_points[idx].name = str(idx)
                        human.left_hand_key_points[idx].x = lm.x
                        human.left_hand_key_points[idx].y = lm.y
                        human.left_hand_key_points[idx].z = lm.z
                        if hand_count == 0:
                            human.right_hand_key_points[idx].name = str(idx)
                            human.right_hand_key_points[idx].x = 0.0
                            human.right_hand_key_points[idx].y = 0.0
                            human.right_hand_key_points[idx].z = 0.0
                hand_count += 1
            human_list.human_hand_list = human
            human_list.num_humans = hand_count
        else:
            # No hands: zero keypoints
            for idx in range(len(self.mp_hands.HandLandmark)):
                human.right_hand_key_points[idx].name = str(idx)
                human.right_hand_key_points[idx].x = 0.0
                human.right_hand_key_points[idx].y = 0.0
                human.right_hand_key_points[idx].z = 0.0
                human.left_hand_key_points[idx].name = str(idx)
                human.left_hand_key_points[idx].x = 0.0
                human.left_hand_key_points[idx].y = 0.0
                human.left_hand_key_points[idx].z = 0.0
            human_list.human_hand_list = human
            human_list.num_humans = 0

        # Publish landmarks
        self.landmark_pub.publish(human_list)

        # Detect and publish gesture
        gesture = self.detect_gesture(results.multi_hand_landmarks)
        str_msg = String(data=gesture)
        self.gesture_pub.publish(str_msg)

        # Display for debugging
        cv2.putText(display_image, gesture, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('MediaPipe Hands', display_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HandsPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
