import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import mediapipe as mp
import torch
import torch.nn as nn
import joblib
from collections import deque, Counter
import os
from ament_index_python.packages import get_package_share_directory
import cv2
from rclpy.duration import Duration


# ---------------------
# Residual Block
# ---------------------
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(size, size)
        self.bn2 = nn.BatchNorm1d(size)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


# ---------------------
# GestureNet Model
# ---------------------
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.res_block1 = ResidualBlock(512)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.res_block2 = ResidualBlock(256)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.res_block3 = ResidualBlock(128)
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.fc1(x)
        x = self.res_block2(x)
        x = self.fc2(x)
        x = self.res_block3(x)
        x = self.fc3(x)
        x = self.output_layer(x)
        return x


# ---------------------
# Normalize landmarks
# ---------------------
def normalize_landmarks(landmark_vector):
    lm = torch.tensor(landmark_vector, dtype=torch.float32).view(21, 3)
    mean = lm.mean(dim=0, keepdim=True)
    lm = lm - mean
    max_val = lm.abs().max()
    if max_val > 0:
        lm = lm / max_val
    return lm.view(-1).tolist()


# ---------------------
# Predict gesture
# ---------------------
def predict_gesture(model, label_encoder, landmark_vector, device):
    normalized_landmarks = normalize_landmarks(landmark_vector)
    input_tensor = torch.tensor(normalized_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
    return predicted_label, confidence.item()


# ---------------------
# ROS2 Node
# ---------------------
class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('gesture_recognition_node')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.9995)
        self.declare_parameter('smoothing_window_size', 15)
        self.declare_parameter('publish_interval_frames', 30)

        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.smoothing_window_size = self.get_parameter('smoothing_window_size').value
        self.publish_interval_frames = self.get_parameter('publish_interval_frames').value

        self.prediction_buffer = deque(maxlen=self.smoothing_window_size)

        # Publisher for detected gesture string
        self.gesture_publisher = self.create_publisher(String, 'control_gesture', 10)

        self.frame_count = 0

        share_dir = get_package_share_directory('tello_detect')
        model_path = os.path.join(share_dir, 'models', 'gesture_model.pt')
        encoder_path = os.path.join(share_dir, 'models', 'label_encoder.pkl')

        self.label_encoder = joblib.load(encoder_path)
        num_classes = len(self.label_encoder.classes_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = 63
        self.model = GestureNet(input_size, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'left_image',
            self.image_callback,
            10)

        # Track last image received time
        self.last_image_time = self.get_clock().now()

        # Timer to check if buffer reset is needed every 100ms
        self.create_timer(0.1, self.check_image_timeout)

        self.get_logger().info('Gesture recognition node started and subscribed to left_image.')

    def image_callback(self, msg):
        self.last_image_time = self.get_clock().now()
        self.frame_count += 1

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                landmark_vector = []
                for lm in hand_landmarks.landmark:
                    landmark_vector.extend([lm.x, lm.y, lm.z])

                if len(landmark_vector) == 63:
                    label, conf = predict_gesture(self.model, self.label_encoder, landmark_vector, self.device)
                    if conf > self.confidence_threshold:
                        self.prediction_buffer.append(label)
                    else:
                        self.prediction_buffer.append("")
        else:
            # No hand landmarks detected; you might want to also append empty or do nothing
            pass

        if self.prediction_buffer:
            most_common = Counter(self.prediction_buffer).most_common(1)[0]
            label, count = most_common
            if label != "" and count >= self.smoothing_window_size // 2:
                cv2.putText(image, f'{label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2)

                # Publish every N frames
                if self.frame_count % self.publish_interval_frames == 0:
                    msg = String()
                    msg.data = label
                    self.gesture_publisher.publish(msg)
                    self.get_logger().info(f'Published gesture: {label}')

        cv2.imshow("Gesture Recognition", image)
        cv2.waitKey(1)

    def check_image_timeout(self):
        now = self.get_clock().now()
        if (now - self.last_image_time) > Duration(seconds=1):
            if self.prediction_buffer:
                self.get_logger().info('No image received for 1 second. Resetting prediction buffer.')
                self.prediction_buffer.clear()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

