import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Adjustable variables for hand detection boxes relative to face
BOX_ASPECT_RATIO = 1.0  # Width/Height ratio of the boxes
BOX_SCALE = 1.8         # Scale factor relative to face height
OFFSET_X = 25           # Horizontal offset of the boxes relative to face bbox edges
OFFSET_Y = 0            # Vertical offset of the boxes relative to face bbox top edge

class FaceHandZoneDetector(Node):
    def __init__(self):
        super().__init__('face_hand_zone_detector')
        self.bridge = CvBridge()

        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.annotated_pub = self.create_publisher(Image, 'annotated_image', 10)
        self.status_pub = self.create_publisher(String, 'hand_zone_status', 10)

        # New publishers for left and right hand zone images
        self.left_image_pub = self.create_publisher(Image, 'left_image', 10)
        self.right_image_pub = self.create_publisher(Image, 'right_image', 10)

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image = cv2.flip(image, 1)
        image2 = image.copy() 
        ih, iw, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("RGB Input", image)
        cv2.waitKey(1)

        face_results = self.face_detector.process(rgb)
        hand_results = self.hand_detector.process(rgb)

        hand_zone_left = None
        hand_zone_right = None

        # Detect face and draw face bounding box + define hand detection zones (left & right)
        if face_results.detections:
            det = face_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # Draw face bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            zone_h = int(h * BOX_SCALE)
            zone_w = int(zone_h * BOX_ASPECT_RATIO)

            # Left box: to the left of the face bbox
            zone_x_left = max(x - zone_w - OFFSET_X, 0)
            zone_y_left = max(0, min(y + OFFSET_Y, ih - zone_h))
            hand_zone_left = (zone_x_left, zone_y_left, zone_w, zone_h)
            cv2.rectangle(image, (zone_x_left, zone_y_left), (zone_x_left + zone_w, zone_y_left + zone_h), (255, 0, 0), 2)
            cv2.putText(image, 'Left Hand Zone', (zone_x_left, zone_y_left - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Right box: to the right of the face bbox
            zone_x_right = min(x + w + OFFSET_X, iw - zone_w)
            zone_y_right = max(0, min(y + OFFSET_Y, ih - zone_h))
            hand_zone_right = (zone_x_right, zone_y_right, zone_w, zone_h)
            cv2.rectangle(image, (zone_x_right, zone_y_right), (zone_x_right + zone_w, zone_y_right + zone_h), (0, 0, 255), 2)
            cv2.putText(image, 'Right Hand Zone', (zone_x_right, zone_y_right - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        hand_in_left_zone = False
        hand_in_right_zone = False

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

                x_vals = [int(lm.x * iw) for lm in landmarks.landmark]
                y_vals = [int(lm.y * ih) for lm in landmarks.landmark]
                x_min, x_max = max(min(x_vals) - 20, 0), min(max(x_vals) + 20, iw)
                y_min, y_max = max(min(y_vals) - 20, 0), min(max(y_vals) + 20, ih)

                hand_area = (x_max - x_min) * (y_max - y_min)

                hand_label = handedness.classification[0].label  # 'Left' or 'Right'

                # Left box: only left hand
                if hand_label == 'Left' and hand_zone_left:
                    zx, zy, zw, zh = hand_zone_left
                    ix_min = max(x_min, zx)
                    iy_min = max(y_min, zy)
                    ix_max = min(x_max, zx + zw)
                    iy_max = min(y_max, zy + zh)
                    intersection_area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
                    if hand_area > 0 and intersection_area / hand_area >= 0.7:
                        hand_in_left_zone = True

                # Right box: any hand
                if hand_zone_right:
                    zx, zy, zw, zh = hand_zone_right
                    ix_min = max(x_min, zx)
                    iy_min = max(y_min, zy)
                    ix_max = min(x_max, zx + zw)
                    iy_max = min(y_max, zy + zh)
                    intersection_area = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
                    if hand_area > 0 and intersection_area / hand_area >= 0.7:
                        hand_in_right_zone = True

                # Draw bounding box around detected hand
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        status_msg = String()
        if hand_in_left_zone and hand_in_right_zone:
            status_msg.data = "Hand in both zones"
        elif hand_in_left_zone:
            status_msg.data = "Left hand in left zone"
        elif hand_in_right_zone:
            status_msg.data = "Hand in right zone"
        else:
            status_msg.data = "Hand outside zones"
        self.status_pub.publish(status_msg)

        # Show status on image
        cv2.putText(image, status_msg.data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if hand_in_left_zone or hand_in_right_zone else (0, 0, 255), 2)

        annotated_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        self.annotated_pub.publish(annotated_msg)
        cv2.imshow("annotated", image)
        cv2.waitKey(1)


        # Publish left or right hand zone images if conditions met
        if hand_in_left_zone:
            left_img_msg = self.bridge.cv2_to_imgmsg(image2, encoding='bgr8')
            self.left_image_pub.publish(left_img_msg)
        if hand_in_right_zone:
            right_img_msg = self.bridge.cv2_to_imgmsg(image2, encoding='bgr8')
            self.right_image_pub.publish(right_img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceHandZoneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

