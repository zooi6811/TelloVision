import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import time
import numpy as np

class FaceTrackingNode(Node):
    def __init__(self):
        super().__init__('face_tracking_node')
        self.bridge = CvBridge()
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4, model_selection=1)

        self.sub_image = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.sub_select = self.create_subscription(String, '/face_enable', self.select_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/control', 10)

        self.width, self.height = 640, 480
        self.xPID = [0.25, 0.0, 0.15]   # yaw
        self.yPID = [0.27, 0.0, 0.1]   # up/down
        self.zPID = [0.0045, 0.0, 0.1] # forward/backward

        self.xTarget = self.width // 2
        self.yTarget = self.height // 2
        self.zTarget = 11500  # Target face area

        # Per-axis PID state
        self.pErrorX, self.Ix, self.pTimeX = 0, 0, time.time()
        self.pErrorY, self.Iy, self.pTimeY = 0, 0, time.time()
        self.pErrorZ, self.Iz, self.pTimeZ = 0, 0, time.time()

        # Activation flags
        self.active = False
        self.prev_active = False

    def select_callback(self, msg):
        self.prev_active = self.active
        self.active = (msg.data == 'face_tracking')

        if self.prev_active and not self.active:
            twist = Twist()  # Zero velocity
            self.cmd_pub.publish(twist)
            self.get_logger().info("Deactivated face tracking — stopping drone.")

    def pid_controller(self, PID, target, cVal, pError, I, pTime, limit=[-100, 100]):
        now = time.time()
        dt = now - pTime
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        error = target - cVal
        P = PID[0] * error
        I += PID[1] * error * dt
        D = PID[2] * (error - pError) / dt
        val = float(np.clip(P + I + D, limit[0], limit[1]))
        return val, error, I, now

    def image_callback(self, msg):
        if not self.active:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        twist = Twist()
        status_text = "Searching for face..."

        if results.detections:
            det = results.detections[0]
            box = det.location_data.relative_bounding_box
            w, h = int(box.width * self.width), int(box.height * self.height)
            x, y = int(box.xmin * self.width), int(box.ymin * self.height)
            cx, cy = x + w // 2, y + h // 2
            area = w * h

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

            # PID for yaw (x)
            xVal, self.pErrorX, self.Ix, self.pTimeX = self.pid_controller(
                self.xPID, self.xTarget, cx, self.pErrorX, self.Ix, self.pTimeX)

            # PID for altitude (y)
            yVal, self.pErrorY, self.Iy, self.pTimeY = self.pid_controller(
                self.yPID, self.yTarget, cy, self.pErrorY, self.Iy, self.pTimeY)

            # PID for forward/backward (z)
            zVal, self.pErrorZ, self.Iz, self.pTimeZ = self.pid_controller(
                self.zPID, self.zTarget, area, self.pErrorZ, self.Iz, self.pTimeZ, limit=[-25, 25])

            twist.angular.z = -xVal
            twist.linear.z = yVal
            twist.linear.y = zVal

            status_text = f"Tracking: X={cx}, Y={cy}, Area={area}"

        self.cmd_pub.publish(twist)
        cv2.putText(frame, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.imshow('Face Tracking', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FaceTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

