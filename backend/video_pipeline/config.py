import torch

# --- Device Configuration ---
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# --- Model Paths ---
YOLO_MODEL_PATH = "model/weights/player_model.pt"
CNN_MODEL_PATH = "model/weights/action_model.pt"

# --- Inference Thresholds ---
CONF = 0.5
IOU = 0.5
CONFIDENCE_THRESHOLD = 0.85 

# --- Crop Dimensions ---
MIN_W = 50
MIN_H = 100
CNN_INPUT_SIZE = (146, 32) 

# --- Classes ---
ACTION_CLASSES = ["spike", "set", "serve", "defense", "block"]