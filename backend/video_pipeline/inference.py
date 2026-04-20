from ultralytics import YOLO
import cv2
import os
import torch
import json
from model.transformations import val_transform
from model.cnn import CNN

# --- GLOBAL CONFIGURATION & MODEL LOADING ---
# We load models globally so they only load ONCE when the Flask server starts.
CONF        = 0.5
IOU         = 0.5
MIN_W       = 50
MIN_H       = 100
CONFIDENCE_THRESHOLD = 0.99
ACTION_CLASSES = ["spike", "set", "serve", "defense", "block"]

_device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# Load YOLO Model
player_model = YOLO("yolov8m.pt")

# Load CNN Action Model
checkpoint = torch.load("model/weights/action_model.pt", map_location=_device)
_num_classes = checkpoint["network.19.weight"].shape[0]
_action_model = CNN(num_classes=_num_classes).to(_device)

# Initialize LazyLinear with a dummy forward pass before loading weights
_dummy = torch.zeros(1, 3, 146, 32).to(_device)
_ = _action_model(_dummy)
_action_model.load_state_dict(checkpoint)
_action_model.eval()

# --- HELPER FUNCTIONS ---

def classify_action(tensor: torch.Tensor) -> str | None:
    tensor = tensor.to(_device)
    with torch.no_grad():
        logits = _action_model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    if conf.item() < CONFIDENCE_THRESHOLD:
        return None
    return ACTION_CLASSES[pred.item()]

def crop_player(frame_bgr, x1, y1, x2, y2):
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = val_transform(crop_rgb)
    return tensor.unsqueeze(0)

def draw_label(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    padding = 4
    bg_x1 = x1
    bg_y1 = max(y1 - text_h - padding * 2, 0)
    bg_x2 = x1 + text_w + padding * 2
    bg_y2 = y1
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(frame, label, (bg_x1 + padding, bg_y2 - padding), font, font_scale, (255, 255, 255), thickness)

def deduplicate_actions(actions, gap_frames=15):
    if not actions:
        return []
    events = []
    prev_frame, prev_action, prev_ts = actions[0]
    start_frame, start_ts = prev_frame, prev_ts

    for frame_idx, action, ts in actions[1:]:
        same_action = (action == prev_action)
        close_in_time = (frame_idx - prev_frame) <= gap_frames

        if same_action and close_in_time:
            prev_frame, prev_ts = frame_idx, ts
        else:
            events.append({
                "action": prev_action,
                "start_ts": start_ts,
                "end_ts": prev_ts,
                "start_frame": start_frame,
                "end_frame": prev_frame
            })
            start_frame, start_ts = frame_idx, ts
            prev_frame, prev_action, prev_ts = frame_idx, action, ts

    events.append({
        "action": prev_action,
        "start_ts": start_ts,
        "end_ts": prev_ts,
        "start_frame": start_frame,
        "end_frame": prev_frame
    })
    return events

# --- MAIN INFERENCE PIPELINE ---

def run_video_inference(input_path: str, output_path: str):
    """Processes a video and saves the annotated version."""
    actions = []
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"Processing {total} frames at {fps:.1f} fps → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps if fps else 0.0

        results = player_model(frame, conf=CONF, iou=IOU, classes=[0], verbose=False)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                if w < MIN_W or h < MIN_H:
                    continue

                tensor = crop_player(frame, x1, y1, x2, y2)
                action = classify_action(tensor)

                if action is not None:
                    actions.append((frame_idx, action, round(timestamp_sec, 2)))
                    draw_label(frame, x1, y1, x2, y2, action.upper(), (0, 255, 0))
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        out.write(frame)
        frame_idx += 1

    events = deduplicate_actions(sorted(actions, key=lambda x: x[0]))

    cap.release()
    out.release()

    return events