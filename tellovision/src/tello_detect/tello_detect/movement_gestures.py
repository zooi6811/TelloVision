import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import mediapipe as mp
from rclpy.duration import Duration
from collections import deque, Counter

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_JOINT_SETS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}

class FaceHandTrackerNode(Node):
    def __init__(self):
        super().__init__('face_hand_tracker_node')
        self.publisher_ = self.create_publisher(Image, 'annotated_image2', 10)
        self.gesture_publisher_ = self.create_publisher(String, 'movement_gesture', 10)
        self.last_gesture = "None"
        self.gesture_buffer = deque(maxlen=10)

        self.bridge = CvBridge()
        self.frame_count = 0

        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.4)

        self.image_subscriber = self.create_subscription(
            Image,
            'right_image',
            self.image_callback,
            10
        )

        self.last_frame_time = self.get_clock().now()
        self.timeout_duration = Duration(seconds=2.0)
        self.timeout_timer = self.create_timer(0.5, self.check_frame_timeout)

    def image_callback(self, msg):
        self.last_frame_time = self.get_clock().now()
        self.frame_count += 1

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("move", image)
        cv2.waitKey(1)
        ih, iw, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        self.face_detection.process(image_rgb)
        hand_results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                self.process_hand(image, hand_landmarks.landmark, iw, ih)

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.publisher_.publish(image_msg)

        if self.frame_count >= 30:
            majority_gesture = self.get_majority_gesture()
            msg = String()
            msg.data = majority_gesture
            self.gesture_publisher_.publish(msg)
            # self.get_logger().info(f"Published gesture: {msg.data}")
            self.frame_count = 0

    def check_frame_timeout(self):
        now = self.get_clock().now()
        if now - self.last_frame_time > self.timeout_duration:
            # self.get_logger().warn("No frame received in timeout window. Resetting frame counter.")
            self.frame_count = 0

    def process_hand(self, image, landmarks, iw, ih):
        x_vals = [int(lm.x * iw) for lm in landmarks]
        y_vals = [int(lm.y * ih) for lm in landmarks]
        x_min, x_max = max(min(x_vals) - 20, 0), min(max(x_vals) + 20, iw)
        y_min, y_max = max(min(y_vals) - 20, 0), min(max(y_vals) + 20, ih)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        direction, tilt, distance = self.estimate_hand_orientation_2d(landmarks, iw, ih)
        thumb_dir, thumb_tilt = self.estimate_thumb_orientation_2d(landmarks, iw, ih)
        fingers = self.fingers_status(landmarks)

        gesture = self.gesture_recognise(direction, thumb_dir, tilt, thumb_tilt, fingers)
        self.last_gesture = gesture
        self.gesture_buffer.append(gesture)

        cv2.putText(image, f"Dir: {direction}, Tilt: {tilt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Thumb: {thumb_dir}, {thumb_tilt}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"Dist: {distance:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(image, f"Fingers: {''.join(map(str, fingers))}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.putText(image, f"Gesture: {gesture}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    def get_majority_gesture(self):
        if not self.gesture_buffer:
            return "None"
        return Counter(self.gesture_buffer).most_common(1)[0][0]

    def estimate_hand_orientation_2d(self, lm, iw, ih):
        x5, y5 = lm[5].x * iw, lm[5].y * ih
        x17, y17 = lm[17].x * iw, lm[17].y * ih
        x0, y0 = lm[0].x * iw, lm[0].y * ih
        x9, y9 = lm[9].x * iw, lm[9].y * ih
        x13, y13 = lm[13].x * iw, lm[13].y * ih
        x_avg = (x5 + x17) / 2
        y_avg = (y5 + y17) / 2
        dist = np.hypot(x9 - x13, y9 - y13) / 20
        dx, dy = x0 - x_avg, y0 - y_avg

        direction = 'right' if dx < (-30 * dist) else 'left' if dx > (30 * dist) else 'centered'
        tilt = 'up' if dy > (30 * dist) else 'down' if dy < (-30 * dist) else 'level'
        return direction, tilt, dist

    def estimate_thumb_orientation_2d(self, lm, iw, ih):
        x0, y0 = lm[0].x * iw, lm[0].y * ih
        x2, y2 = lm[2].x * iw, lm[2].y * ih
        x9, y9 = lm[9].x * iw, lm[9].y * ih
        x13, y13 = lm[13].x * iw, lm[13].y * ih
        dist = np.hypot(x9 - x13, y9 - y13) / 20
        dx, dy = x0 - x2, y0 - y2

        direction = 'left' if dx > (30 * dist) else 'right' if dx < (-30 * dist) else 'neutral'
        tilt = 'up' if dy > (30 * dist) else 'down' if dy < (-30 * dist) else 'level'
        return direction, tilt

    def fingers_status(self, lm):
        status = []
        for finger, joints in FINGER_JOINT_SETS.items():
            threshold = 30 if finger == "Thumb" else 20
            pts = [np.array([lm[i].x, lm[i].y, lm[i].z]) for i in joints]
            vecs = [pts[i+1] - pts[i] for i in range(len(pts)-1)]
            vecs = [v / np.linalg.norm(v) for v in vecs]
            straight = all(np.degrees(np.arccos(np.clip(np.dot(vecs[i], vecs[i+1]), -1.0, 1.0))) < threshold
                           for i in range(len(vecs)-1))
            status.append(1 if straight else 0)
        return status

    def gesture_recognise(self, hand_direction, thumb_direction, hand_tilt, thumb_tilt, fingers):
        gesture = "Unknown"
        if fingers == [0, 1, 1, 0, 0]:
            if hand_tilt == "up":
            	gesture = "Peace"
            elif hand_tilt == "left":
            	gesture = "Kawaii"
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "Open_Palm"
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = "Fist"
        elif fingers == [0, 1, 0, 0, 1] and hand_tilt == "up":
            gestures = "Spider"
        elif fingers == [1, 0, 0, 0, 0]:
            if thumb_tilt == "up":
                gesture = "Thumbs_Up"
            elif thumb_tilt == "down":
                gesture = "Thumbs_Down"
            elif thumb_tilt == "right":
            	gesture = "Thumbs_Right"
            elif thumb_tilt == "left":
            	gesture = "Thumbs_Left"
        elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            if hand_tilt == "up":
                gesture = "Point_Up"
            elif hand_tilt == "down":
                gesture = "Point_Down"
            elif hand_direction == "left":
                gesture = "Point_Left"
            elif hand_direction == "right":
                gesture = "Point_Right"
        elif fingers == [0, 0, 1, 0, 0] or fingers == [1, 0, 1, 0, 0,]:
            if hand_tilt == "up":
                gesture = "Assert_Dominance"
        return gesture

def main(args=None):
    rclpy.init(args=args)
    node = FaceHandTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

