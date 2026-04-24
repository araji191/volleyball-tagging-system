import torch

# --- Device Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Model Paths ---
YOLO_MODEL_PATH = "/Users/abiolaraji/Desktop/volleyball-tagging-system/backend/model/weights/yolov8x-pose.pt"
POSE_CLASSIFIER_PATH = "/Users/abiolaraji/Desktop/volleyball-tagging-system/backend/model/weights/pose_classifier_best.pt"

# --- Inference Thresholds ---
CONF = 0.5               # YOLO detection confidence
IOU = 0.5                # YOLO NMS threshold
CONFIDENCE_THRESHOLD = 0.50  # Classifier softmax threshold
GAP_FRAMES = 15          # Deduplication window
TOP_N_PLAYERS = 3

# --- Crop/Box Dimensions ---
MIN_W = 50
MIN_H = 100

# --- Classes ---
CLASS_MAP = {
    "spike":   0,
    "defense": 1,
    "block":   2,
    "set":     3,
    "serve":   4,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
ACTION_CLASSES = list(CLASS_MAP.keys())