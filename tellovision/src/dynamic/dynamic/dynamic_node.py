#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String as RosStringMsg
from cv_bridge import CvBridge, CvBridgeError

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, Counter
import os
# import time # Not explicitly used in this version, can be removed if not needed later

# --- Configuration for Dynamic Gesture Model ---

# MODIFIED: Gestures the new model was trained on (MUST match your training script)
MODEL_TRAINING_GESTURES = np.array(['swipe_left', 'swipe_right', 'circle', 'wave', 'swipe_vert'])
NUM_CLASSES_FOR_MODEL_INIT = len(MODEL_TRAINING_GESTURES) # This will now be 5

# Gestures for application-level logic and final output (can be different)
# This example maps swipe_left/right to swipe_side, and keeps others.
# If your new model directly outputs 'swipe_vert' and you want to use it, add it here
# or adjust the mapping logic in DynamicGestureProcessor.
GESTURES_APP_LEVEL = np.array(['swipe_side', 'circle', 'wave', 'swipe_vert']) # Ensure this includes 'swipe_vert' if used directly

# MODIFIED: Feature engineering constants (MUST match your training script)
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK_NORMALIZED = 3 # x, y, z for hand-normalized

NUM_HAND_NORMALIZED_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK_NORMALIZED # 21 * 3 = 63
NUM_RAW_WRIST_FEATURES = 3 # Raw x, y, z of the wrist in image coordinates
RAW_FEATURES_PER_FRAME_EXTRACTION = NUM_HAND_NORMALIZED_FEATURES + NUM_RAW_WRIST_FEATURES # 63 + 3 = 66

MODEL_INPUT_FEATURES = RAW_FEATURES_PER_FRAME_EXTRACTION * 2 # (63+3)*2 = 132 for pos+vel
MODEL_SEQUENCE_LENGTH = SEQUENCE_LENGTH - 1
# --- End Feature Constants ---


# --- Model and Normalization File Paths ---
MODEL_FILENAME = 'dynamic_gesture_model_best_loss.pth' # Default, ensure this is your NEW model's name
NORMALIZATION_PARAMS_FILENAME = 'normalization_params.npz' # Saved by new training script

_MODEL_PATH_FINAL = MODEL_FILENAME # Default if not found elsewhere
_NORM_PARAMS_PATH_FINAL = NORMALIZATION_PARAMS_FILENAME # Default

try:
    from ament_index_python.packages import get_package_share_directory
    PACKAGE_NAME_FOR_MODEL = 'dynamic' # Make this your ROS package name
    package_share_dir = get_package_share_directory(PACKAGE_NAME_FOR_MODEL)
    
    _model_path_in_share = os.path.join(package_share_dir, 'models', MODEL_FILENAME)
    _norm_params_path_in_share = os.path.join(package_share_dir, 'models', NORMALIZATION_PARAMS_FILENAME)

    if os.path.exists(_model_path_in_share):
        _MODEL_PATH_FINAL = _model_path_in_share
        print(f"INFO: Using dynamic gesture model from package share: {_MODEL_PATH_FINAL}")
    else:
        print(f"INFO: Model not found in package '{PACKAGE_NAME_FOR_MODEL}' share dir ('{_model_path_in_share}'). Checking local paths.")

    if os.path.exists(_norm_params_path_in_share):
        _NORM_PARAMS_PATH_FINAL = _norm_params_path_in_share
        print(f"INFO: Using normalization params from package share: {_NORM_PARAMS_PATH_FINAL}")
    else:
        print(f"INFO: Normalization params not found in package '{PACKAGE_NAME_FOR_MODEL}' share dir ('{_norm_params_path_in_share}'). Checking local paths.")

except (ImportError, ModuleNotFoundError, Exception) as e: # Added ModuleNotFoundError for ROS2 Humble/ament
    print(f"INFO: ament_index_python or package not found (error: {type(e).__name__}: {e}). Will attempt to load files from local script directory or 'models' subdirectory.")
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    
    _model_path_local_script = os.path.join(_script_dir, MODEL_FILENAME)
    _model_path_local_models_dir = os.path.join(_script_dir, 'models', MODEL_FILENAME)
    
    _norm_params_local_script = os.path.join(_script_dir, NORMALIZATION_PARAMS_FILENAME)
    _norm_params_local_models_dir = os.path.join(_script_dir, 'models', NORMALIZATION_PARAMS_FILENAME)

    if os.path.exists(_model_path_local_script):
        _MODEL_PATH_FINAL = _model_path_local_script
        print(f"INFO: Using dynamic gesture model from script directory: {_MODEL_PATH_FINAL}")
    elif os.path.exists(_model_path_local_models_dir):
        _MODEL_PATH_FINAL = _model_path_local_models_dir
        print(f"INFO: Using dynamic gesture model from local 'models' directory: {_MODEL_PATH_FINAL}")
    else:
        print(f"WARNING: Dynamic gesture model file '{MODEL_FILENAME}' not found in package share, script directory, or local 'models' subdirectory.")

    if os.path.exists(_norm_params_local_script):
        _NORM_PARAMS_PATH_FINAL = _norm_params_local_script
        print(f"INFO: Using normalization params from script directory: {_NORM_PARAMS_PATH_FINAL}")
    elif os.path.exists(_norm_params_local_models_dir):
        _NORM_PARAMS_PATH_FINAL = _norm_params_local_models_dir
        print(f"INFO: Using normalization params from local 'models' directory: {_NORM_PARAMS_PATH_FINAL}")
    else:
        print(f"WARNING: Normalization params file '{NORMALIZATION_PARAMS_FILENAME}' not found in package share, script directory, or local 'models' subdirectory.")


MODEL_PATH = _MODEL_PATH_FINAL
NORMALIZATION_PARAMS_PATH = _NORM_PARAMS_PATH_FINAL

# LSTM Model Hyperparameters (MUST match training)
HIDDEN_SIZE1 = 128; HIDDEN_SIZE2 = 256
DROPOUT_RATE_MODEL = 0.4; BIDIRECTIONAL_MODEL = True; USE_ATTENTION_MODEL = True

# Dynamic Gesture Prediction Parameters
PREDICTION_THRESHOLD = 0.65 # You might need to adjust this with the new model
PREDICTION_INTERVAL_FRAMES = 2
SMOOTHING_WINDOW_SIZE = 3
MOTION_THRESHOLD = 0.005 # May need re-tuning

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dynamic Gesture Model Definition (LSTM) --- (Keep as is, NUM_CLASSES_FOR_MODEL_INIT handles it)
class Attention(nn.Module):
    def __init__(self, hidden_size): super(Attention, self).__init__(); self.attention_weights_layer = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, lstm_output): energies = self.attention_weights_layer(lstm_output).squeeze(-1); attn_weights = F.softmax(energies, dim=1); context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1); return context_vector, attn_weights

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout_rate, bidirectional_lstm, use_attention):
        super(GestureLSTM, self).__init__(); self.use_attention = use_attention
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=bidirectional_lstm); self.dropout1 = nn.Dropout(dropout_rate)
        lstm2_input_size = hidden_size1 * (2 if bidirectional_lstm else 1)
        self.lstm2 = nn.LSTM(lstm2_input_size, hidden_size2, batch_first=True, bidirectional=bidirectional_lstm); self.dropout2 = nn.Dropout(dropout_rate)
        fc_input_from_lstm_size = hidden_size2 * (2 if bidirectional_lstm else 1)
        if self.use_attention: self.attention_layer = Attention(fc_input_from_lstm_size)
        self.fc1 = nn.Linear(fc_input_from_lstm_size, 64); self.relu1 = nn.ReLU(); self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32); self.relu2 = nn.ReLU(); self.fc3 = nn.Linear(32, num_classes)
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x); lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1); lstm_out2 = self.dropout2(lstm_out2)
        if self.use_attention: final_lstm_representation, _ = self.attention_layer(lstm_out2)
        else: final_lstm_representation = lstm_out2[:, -1, :]
        out = self.fc1(final_lstm_representation); out = self.relu1(out); out = self.dropout3(out)
        out = self.fc2(out); out = self.relu2(out); out = self.fc3(out); return out

# --- MODIFIED: Helper functions for Dynamic Gesture Recognition ---
def _extract_combined_features_from_landmarks(landmarks_mp_list):
    # landmarks_mp_list is results.multi_hand_landmarks[0].landmark
    output_features_frame = np.zeros(RAW_FEATURES_PER_FRAME_EXTRACTION)
    if not landmarks_mp_list or len(landmarks_mp_list) != NUM_LANDMARKS:
        return output_features_frame # Return zeros if no valid landmarks

    # Convert to NumPy array for easier processing
    # landmarks_mp_list elements have .x, .y, .z
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_mp_list])

    # 1. Hand-normalized features
    wrist_coords_raw_for_norm = landmarks_array[0, :].copy()
    hand_normalized_landmarks = landmarks_array - wrist_coords_raw_for_norm
    middle_finger_mcp_coords_relative = hand_normalized_landmarks[9, :]
    hand_size = np.linalg.norm(middle_finger_mcp_coords_relative)
    if hand_size > 1e-6:
        hand_normalized_landmarks = hand_normalized_landmarks / hand_size
    hand_normalized_flat = hand_normalized_landmarks.flatten()
    output_features_frame[:NUM_HAND_NORMALIZED_FEATURES] = hand_normalized_flat

    # 2. Raw (image-normalized) wrist coordinates from MediaPipe
    # landmarks_mp_list[0] corresponds to the wrist landmark
    raw_wrist_x = landmarks_mp_list[0].x
    raw_wrist_y = landmarks_mp_list[0].y
    raw_wrist_z = landmarks_mp_list[0].z
    output_features_frame[NUM_HAND_NORMALIZED_FEATURES:] = np.array([raw_wrist_x, raw_wrist_y, raw_wrist_z])
    
    return output_features_frame

def create_pos_vel_features_for_sequence(raw_feature_sequence_np):
    # raw_feature_sequence_np shape: (SEQUENCE_LENGTH, RAW_FEATURES_PER_FRAME_EXTRACTION)
    if raw_feature_sequence_np.ndim != 2 or \
       raw_feature_sequence_np.shape[0] != SEQUENCE_LENGTH or \
       raw_feature_sequence_np.shape[1] != RAW_FEATURES_PER_FRAME_EXTRACTION:
        return None

    raw_sequences_batch_mock = raw_feature_sequence_np[np.newaxis, :, :]

    hand_norm_features_raw = raw_sequences_batch_mock[:, :, :NUM_HAND_NORMALIZED_FEATURES]
    raw_wrist_features_raw   = raw_sequences_batch_mock[:, :, NUM_HAND_NORMALIZED_FEATURES:]

    hand_norm_velocities = np.diff(hand_norm_features_raw, axis=1)
    raw_wrist_velocities   = np.diff(raw_wrist_features_raw, axis=1)
    
    hand_norm_positions_concat = hand_norm_features_raw[:, 1:, :]
    raw_wrist_positions_concat   = raw_wrist_features_raw[:, 1:, :]
    
    final_features_batch_mock = np.concatenate(
        (hand_norm_positions_concat, hand_norm_velocities,
         raw_wrist_positions_concat, raw_wrist_velocities),
        axis=2
    )
    return final_features_batch_mock.squeeze(0) # Shape: (MODEL_SEQUENCE_LENGTH, MODEL_INPUT_FEATURES)

# --- Dynamic Gesture Processor Class ---
class DynamicGestureProcessor:
    def __init__(self, model_path_param, norm_params_path_param, node_logger): # Added norm_params_path_param
        self.logger = node_logger
        self.model = None
        self.train_mean = None
        self.train_std = None
        self.epsilon = 1e-7 # For normalization

        self._load_model_and_norm_params(model_path_param, norm_params_path_param)

        if self.model is None or self.train_mean is None or self.train_std is None:
            self.logger.fatal("Dynamic Gesture Model or Normalization Params failed to load. Dynamic gesture recognition will be disabled.")
        
        self.landmark_sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) # Stores RAW_FEATURES_PER_FRAME_EXTRACTION
        self.current_internal_status = "No Dynamic Gesture"
        self.recent_predictions_for_smoothing = deque(maxlen=SMOOTHING_WINDOW_SIZE)
        self.frame_counter_for_internal_prediction = 0
        self.internal_states_in_publish_window = []
        self.logger.info(f"DynamicGestureProcessor initialised. Device: {DEVICE}. Smoothing: {SMOOTHING_WINDOW_SIZE}, Interval: {PREDICTION_INTERVAL_FRAMES} frames.")
        self.logger.info(f"Model configured for {NUM_CLASSES_FOR_MODEL_INIT} classes: {list(MODEL_TRAINING_GESTURES)}")
        self.logger.info(f"Application will map to gestures: {list(GESTURES_APP_LEVEL)}")

    def _load_model_and_norm_params(self, model_p, norm_p):
        if not os.path.exists(model_p):
            self.logger.error(f"Dynamic gesture model file not found at {model_p}"); return
        if not os.path.exists(norm_p):
            self.logger.error(f"Normalization params file not found at {norm_p}"); return

        # Load Model
        self.model = GestureLSTM(MODEL_INPUT_FEATURES, HIDDEN_SIZE1, HIDDEN_SIZE2,
                                 NUM_CLASSES_FOR_MODEL_INIT, # Use the count of gestures model was trained on
                                 DROPOUT_RATE_MODEL, BIDIRECTIONAL_MODEL, USE_ATTENTION_MODEL)
        try:
            self.model.load_state_dict(torch.load(model_p, map_location=DEVICE))
            self.logger.info(f"Dynamic gesture LSTM model loaded successfully from {model_p}")
            self.model.to(DEVICE); self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading dynamic gesture LSTM model state_dict: {e}"); self.model = None; return

        # Load Normalization Parameters
        try:
            norm_data = np.load(norm_p)
            self.train_mean = norm_data['mean']
            self.train_std = norm_data['std']
            # Verify feature dimension
            if self.train_mean.shape[-1] != MODEL_INPUT_FEATURES or self.train_std.shape[-1] != MODEL_INPUT_FEATURES:
                self.logger.error(f"Normalization param feature size ({self.train_mean.shape[-1]}) mismatch with MODEL_INPUT_FEATURES ({MODEL_INPUT_FEATURES}).")
                self.train_mean = self.train_std = self.model = None # Invalidate all
                return
            self.logger.info(f"Normalization parameters loaded successfully from {norm_p} (Features: {self.train_mean.shape[-1]})")
        except Exception as e:
            self.logger.error(f"Error loading normalization parameters: {e}"); self.train_mean = self.train_std = self.model = None; return


    def process_landmarks(self, hand_landmarks_mp_list, is_hand_considered_active, trigger_publish_decision: bool = False):
        final_gesture_to_publish_for_ros = None
        if self.model is None or self.train_mean is None or self.train_std is None: # Check all loaded
            self.current_internal_status = "Model/Norm Error"
            if trigger_publish_decision: self.internal_states_in_publish_window = []
            return self.current_internal_status, None

        # Use _extract_combined_features_from_landmarks
        active_pipeline_features = _extract_combined_features_from_landmarks(hand_landmarks_mp_list if is_hand_considered_active else None)
        current_cycle_determined_state = "no_gesture" # Default state for the current processing cycle

        if not is_hand_considered_active:
            self.current_internal_status = "No Hand Detected"
            self.recent_predictions_for_smoothing.clear()
            current_cycle_determined_state = "no_hand_detected_dg"
        elif not np.any(active_pipeline_features): # If zeros were returned due to invalid landmark list length
             self.current_internal_status = "Invalid Landmarks" # Or "No Hand" if more appropriate
             self.recent_predictions_for_smoothing.clear()
             current_cycle_determined_state = "no_gesture" # Treat as no gesture
        else:
            if self.current_internal_status in ["No Hand Detected", "No Dynamic Gesture", "Model/Norm Error", "Invalid Landmarks"]:
                self.current_internal_status = "Processing..."
        
        self.landmark_sequence_buffer.append(active_pipeline_features) # Appends RAW_FEATURES_PER_FRAME_EXTRACTION

        if len(self.landmark_sequence_buffer) == SEQUENCE_LENGTH:
            self.frame_counter_for_internal_prediction += 1
            if self.frame_counter_for_internal_prediction >= PREDICTION_INTERVAL_FRAMES:
                self.frame_counter_for_internal_prediction = 0
                
                raw_feature_sequence_np = np.array(list(self.landmark_sequence_buffer), dtype=np.float32)

                if not is_hand_considered_active : # Re-check if hand became inactive during buffer fill
                    self.current_internal_status = "No Hand Detected"
                    self.recent_predictions_for_smoothing.clear()
                    current_cycle_determined_state = "no_hand_detected_dg"
                # Check if buffer is all zeros (e.g. if hand was lost for the whole buffer duration)
                elif not np.any(raw_feature_sequence_np): # If all frames in buffer had no hand
                    self.current_internal_status = "No Hand (Buffer Empty)"
                    self.recent_predictions_for_smoothing.clear()
                    current_cycle_determined_state = "no_gesture"
                else:
                    # Motion metric on hand-normalized part only
                    hand_norm_part_for_motion = raw_feature_sequence_np[:, :NUM_HAND_NORMALIZED_FEATURES]
                    motion_metric = np.sum(np.var(hand_norm_part_for_motion.reshape(SEQUENCE_LENGTH, NUM_LANDMARKS, FEATURES_PER_LANDMARK_NORMALIZED), axis=0))
                    
                    if motion_metric < MOTION_THRESHOLD:
                        self.current_internal_status = "Static Hand"
                        self.recent_predictions_for_smoothing.clear()
                        current_cycle_determined_state = "static_hand_dg"
                    else:
                        model_input_unnormalized = create_pos_vel_features_for_sequence(raw_feature_sequence_np)
                        if model_input_unnormalized is not None:
                            # Normalize features before feeding to model
                            model_input_normalized = (model_input_unnormalized - self.train_mean.squeeze()) / (self.train_std.squeeze() + self.epsilon)
                            
                            model_input_tensor = torch.tensor(model_input_normalized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                outputs = self.model(model_input_tensor)
                                probabilities = F.softmax(outputs, dim=1)
                                confidence, predicted_idx = torch.max(probabilities, 1)
                            
                            raw_model_predicted_label = MODEL_TRAINING_GESTURES[predicted_idx.item()]
                            prediction_confidence = confidence.item()
                            
                            # Map model's raw output to application-level gestures
                            application_predicted_label = "Uncertain" # Default
                            if raw_model_predicted_label == 'swipe_left' or raw_model_predicted_label == 'swipe_right':
                                application_predicted_label = 'swipe_side'
                            elif raw_model_predicted_label == 'circle':
                                application_predicted_label = 'circle'
                            elif raw_model_predicted_label == 'wave':
                                application_predicted_label = 'wave'
                            elif raw_model_predicted_label == 'swipe_vert': # Handle new gesture
                                application_predicted_label = 'swipe_vert'
                            # Add other mappings if MODEL_TRAINING_GESTURES and GESTURES_APP_LEVEL differ more

                            if prediction_confidence >= PREDICTION_THRESHOLD:
                                self.recent_predictions_for_smoothing.append(application_predicted_label)
                                current_cycle_determined_state = application_predicted_label
                            else:
                                self.recent_predictions_for_smoothing.append("Uncertain")
                                current_cycle_determined_state = "no_gesture" # Or "uncertain_dg"

                            if len(self.recent_predictions_for_smoothing) >= SMOOTHING_WINDOW_SIZE:
                                counts = Counter(self.recent_predictions_for_smoothing)
                                most_common_prediction, _ = counts.most_common(1)[0]
                                self.current_internal_status = most_common_prediction if most_common_prediction != "Uncertain" else "No Dyn. Gesture"
                        else: # model_input_unnormalized is None
                            self.current_internal_status = "Feature Proc. Error"
                            current_cycle_determined_state = "error_dg"
        else: # Buffer not full
             if not is_hand_considered_active: # Update status if hand lost while buffer filling
                 self.current_internal_status = "No Hand Detected"
                 current_cycle_determined_state = "no_hand_detected_dg"


        self.internal_states_in_publish_window.append(current_cycle_determined_state)

        if trigger_publish_decision:
            determined_publish_gesture = "idle" # Default for ROS message
            
            # Prioritize swipe_side if it occurred recently
            swipe_priority_gesture_found = any(state == 'swipe_side' for state in self.internal_states_in_publish_window)

            if swipe_priority_gesture_found:
                determined_publish_gesture = 'swipe_side'
            else:
                # Consider only valid application-level gestures for voting if no swipe_side
                actual_app_gestures_in_window = [s for s in self.internal_states_in_publish_window if s in GESTURES_APP_LEVEL]
                if actual_app_gestures_in_window:
                    counts = Counter(actual_app_gestures_in_window)
                    determined_publish_gesture = counts.most_common(1)[0][0]
                # If no app-level gestures, it remains "idle"

            if self.model and self.train_mean is not None: # Check if processor is functional
                final_gesture_to_publish_for_ros = determined_publish_gesture
            else: # Model or norm params not loaded
                final_gesture_to_publish_for_ros = "error_model_unavailable" # Or None, depending on desired ROS msg

            self.internal_states_in_publish_window = [] # Reset for next publish window

        return self.current_internal_status, final_gesture_to_publish_for_ros


class TelloGestureControllerNode(Node):
    def __init__(self):
        super().__init__('tello_gesture_controller_node')
        self.get_logger().info(f"Node initialising...")
        self.get_logger().info(f"Attempting to use Gesture Model Path: {MODEL_PATH}")
        self.get_logger().info(f"Attempting to use Norm Params Path: {NORMALIZATION_PARAMS_PATH}")


        image_topic = "/right_image" # Default, consider making this a parameter
        dynamic_gesture_topic = "dynamic_gesture_input"
        
        self.FRAMES_PER_PUBLISH = 30 # Publish decision every X processed frames
        self.processed_frame_count = 0

        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10) # QoS profile 10 is common
        self.dynamic_gesture_publisher = self.create_publisher(
            RosStringMsg, dynamic_gesture_topic, 10)

        self.get_logger().info(f"Subscribed to image topic: '{image_topic}'")
        self.get_logger().info(f"Publishing dynamic gestures to: '{dynamic_gesture_topic}' every {self.FRAMES_PER_PUBLISH} frames.")
        self.get_logger().info(f"Application will output from gestures: {list(GESTURES_APP_LEVEL)}")


        self.mp_hands_solution = mp.solutions.hands # Store the solution module
        self.hands_detector = self.mp_hands_solution.Hands( # Use the stored module
            max_num_hands=1,
            min_detection_confidence=0.4, # Adjusted for potentially faster/more detections
            min_tracking_confidence=0.4
        )

        self.gesture_processor = DynamicGestureProcessor(
            model_path_param=MODEL_PATH,
            norm_params_path_param=NORMALIZATION_PARAMS_PATH, # Pass norm params path
            node_logger=self.get_logger()
        )
        if self.gesture_processor.model is None or \
           self.gesture_processor.train_mean is None or \
           self.gesture_processor.train_std is None:
            self.get_logger().fatal("Gesture processor failed to initialize model/norm_params. Node may not function correctly.")
            # Decide on behavior: shutdown or run without dynamic gestures
            # For now, it will publish "error_model_unavailable" or similar based on processor logic

    def image_callback(self, msg):
        try:
            cv_frame_original = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Flip for a mirror effect, common in gesture UI
        image_display = cv2.flip(cv_frame_original, 1) 

        # Process for hand landmarks
        image_rgb_for_hands = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB) # Use flipped image for consistency
        image_rgb_for_hands.flags.writeable = False # Performance optimization
        hand_results_mp = self.hands_detector.process(image_rgb_for_hands)
        image_rgb_for_hands.flags.writeable = True

        current_hand_landmarks_mp_list = None # This should be the list of landmark objects
        is_hand_considered_active = False

        if hand_results_mp.multi_hand_landmarks:
            # Get the landmark list from the first detected hand
            current_hand_landmarks_mp_list = hand_results_mp.multi_hand_landmarks[0].landmark 
            is_hand_considered_active = True 
        
        self.processed_frame_count += 1
        should_trigger_publish_decision = False
        if self.processed_frame_count >= self.FRAMES_PER_PUBLISH:
            should_trigger_publish_decision = True
            self.processed_frame_count = 0

        # Pass the list of landmark objects
        internal_status, dynamic_gesture_to_publish = self.gesture_processor.process_landmarks(
            current_hand_landmarks_mp_list, 
            is_hand_considered_active,
            trigger_publish_decision=should_trigger_publish_decision
        )

        if dynamic_gesture_to_publish is not None and dynamic_gesture_to_publish != "error_model_unavailable":
            gesture_msg = RosStringMsg()
            gesture_msg.data = dynamic_gesture_to_publish
            self.dynamic_gesture_publisher.publish(gesture_msg)
            self.get_logger().debug( # Changed to debug to reduce console spam, info for actual publishes
                f"Published: '{dynamic_gesture_to_publish}'. Internal: '{internal_status}'"
            )
        elif should_trigger_publish_decision: # Log even if nothing was published on a publish frame
             self.get_logger().debug(
                f"No gesture published. Final Decision: '{dynamic_gesture_to_publish}', Internal Status: '{internal_status}'"
             )
        
        # Optional: Display window for debugging (can be removed for headless operation)
        # cv2.putText(image_display, f"Dyn: {internal_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # if hand_results_mp.multi_hand_landmarks:
        #     for hand_landmarks_instance in hand_results_mp.multi_hand_landmarks:
        #         mp.solutions.drawing_utils.draw_landmarks(
        #             image_display, hand_landmarks_instance, self.mp_hands_solution.HAND_CONNECTIONS)
        # cv2.imshow("Tello Gesture Feed", image_display)
        # cv2.waitKey(1)


    def destroy_node(self):
        self.get_logger().info("Shutting down Tello Gesture Controller Node...")
        if hasattr(self, 'hands_detector') and self.hands_detector:
             self.hands_detector.close()
        # cv2.destroyAllWindows() # If using imshow
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TelloGestureControllerNode()
        # Check if gesture processor initialized correctly after creating the node
        if node.gesture_processor.model is None or \
           node.gesture_processor.train_mean is None or \
           node.gesture_processor.train_std is None:
             node.get_logger().fatal(f"CRITICAL: Gesture model or normalization parameters failed to load. Node will not perform dynamic gestures effectively and will shut down.")
             # Optionally, allow node to run but with dynamic gestures disabled
             # For now, shutting down if critical components are missing.
             rclpy.shutdown()
             return # Exit main if critical components failed
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        if node: node.get_logger().info("Keyboard interrupt received, shutting down.")
    except Exception as e:
        # Log any other exceptions
        if node:
            node.get_logger().fatal(f"Unhandled exception in TelloGestureControllerNode: {e}", exc_info=True)
        else:
            print(f"Unhandled exception before node initialization: {e}")
    finally:
        if node and rclpy.ok(): # Check if node exists and rclpy is still okay
            node.destroy_node()
        if rclpy.ok(): # Ensure shutdown is only called if not already shutting down
            rclpy.shutdown()
    print("INFO: Application terminated.")

if __name__ == '__main__':
    main()
