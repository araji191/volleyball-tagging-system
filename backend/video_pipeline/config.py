"""
This module centralizes all configuration settings for the video pose inference pipeline. 
This is where device selection, model paths, and thresholds are defined.

Authors: Abiola Raji, Patrick Dang 
"""

import torch

# --- Device Configuration ---
# Try Nvidia GPU first, then Apple MPS, then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(DEVICE)

# --- Model Paths ---
# To run update to your own paths should be downloaded off the google drive
# YOLO_MODEL_PATH file name: yolov8x-pose.pt
# POSE_CLASSIFIER_PATH file name: pose_classifier_best.pt
YOLO_MODEL_PATH = r"C:\Users\User\Documents\CODE\volleyball-tagging-system\backend\model\weights\yolov8x-pose.pt"
POSE_CLASSIFIER_PATH = r"C:\Users\User\Documents\CODE\volleyball-tagging-system\backend\model\weights\pose_classifier_best.pt"

# --- Inference Thresholds ---
CONF = 0.5               # YOLO detection confidence
IOU = 0.5                # YOLO NMS threshold
CONFIDENCE_THRESHOLD = 0.50  # Classifier softmax threshold
GAP_FRAMES = 15          # Deduplication window

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