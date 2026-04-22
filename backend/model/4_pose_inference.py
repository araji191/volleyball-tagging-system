import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
INPUT_PATH  = "videos/back_view.mp4"
OUTPUT_PATH = "output/back_view_pose.mp4"

YOLO_MODEL_PATH      = "yolov8s-pose.pt"   # pose model, not detection
POSE_CLASSIFIER_PATH = "pose_classifier_best.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CONF               = 0.5      # YOLO person detection confidence
IOU                = 0.5
CONFIDENCE_THRESHOLD = 0.55   # classifier softmax threshold — below this → no label
MIN_W              = 50       # skip tiny boxes
MIN_H              = 100
GAP_FRAMES         = 15       # deduplicate window

CLASS_MAP = {
    "spike":   0,
    "defense": 1,
    "block":   2,
    "set":     3,
    "serve":   4,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# ----------------------------
# FEATURE ENGINEERING
# (must match train.py exactly)
# ----------------------------
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def normalize_keypoints(kp):
    kp = np.array(kp)
    xy   = kp[:, :2].copy()
    conf = kp[:, 2:]

    xy[conf.squeeze() < 0.3] = 0

    center = (xy[11] + xy[12]) / 2
    xy = xy - center

    torso = np.linalg.norm((xy[5] + xy[6]) / 2)
    if torso > 0:
        xy = xy / torso

    if xy[6][0] < xy[5][0]:
        xy[:, 0] *= -1

    angles = []
    angles.append(get_angle(xy[5],  xy[7],  xy[9]))   # left elbow
    angles.append(get_angle(xy[6],  xy[8],  xy[10]))  # right elbow
    angles.append(get_angle(xy[11], xy[13], xy[15]))  # left knee
    angles.append(get_angle(xy[12], xy[14], xy[16]))  # right knee
    angles.append(get_angle(xy[7],  xy[5],  xy[11]))  # left shoulder
    angles.append(get_angle(xy[8],  xy[6],  xy[12]))  # right shoulder

    return np.concatenate([xy.flatten(), conf.flatten(), np.array(angles)])


# ----------------------------
# MODEL
# ----------------------------
class PoseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(57, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# MODEL LOADING (once at startup)
# ----------------------------
print(f"Loading models on {DEVICE}...")

pose_model = YOLO(YOLO_MODEL_PATH)
try:
    pose_model.to("mps")
except:
    pass

classifier = PoseClassifier().to(DEVICE)
classifier.load_state_dict(torch.load(POSE_CLASSIFIER_PATH, map_location=DEVICE))
classifier.eval()

print("Models loaded ✅")


# ----------------------------
# HELPERS
# ----------------------------
def extract_keypoints_from_result(result):
    """
    Given a single YOLO pose result, returns keypoints for the largest person.
    Returns None if no valid detection.
    """
    if result.keypoints is None:
        return None

    kp_tensor = result.keypoints.data
    if kp_tensor is None or len(kp_tensor) == 0:
        return None

    keypoints = kp_tensor.cpu().numpy()

    # pick largest person by box area
    if len(keypoints) > 1 and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = areas.argmax()
            keypoints = keypoints[idx]
        else:
            keypoints = keypoints[0]
    else:
        keypoints = keypoints[0]

    return keypoints  # shape (17, 3)


def classify_pose(keypoints_raw):
    """
    Takes raw (17, 3) keypoints array, returns (action_str, confidence) or (None, None).
    """
    try:
        features = normalize_keypoints(keypoints_raw)
    except Exception:
        return None, None

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    conf_val = conf.item()
    if conf_val < CONFIDENCE_THRESHOLD:
        return None, conf_val

    return IDX_TO_CLASS[pred.item()], conf_val


def get_boxes_from_result(result):
    """Returns list of (x1, y1, x2, y2) int tuples from a YOLO result."""
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    return [tuple(map(int, box)) for box in boxes]


def draw_label(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label:
        return
    text = f"{label.upper()} {conf:.2f}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad   = 4
    bg_y1 = max(y1 - text_h - pad * 2, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + text_w + pad * 2, y1), color, -1)
    cv2.putText(frame, text, (x1 + pad, y1 - pad), font, font_scale, (255, 255, 255), thickness)


def deduplicate_actions(actions, gap_frames=GAP_FRAMES):
    """Collapses consecutive same-action detections into events."""
    if not actions:
        return []

    events = []
    prev_frame, prev_action, prev_ts = actions[0]
    start_frame, start_ts = prev_frame, prev_ts

    for frame_idx, action, ts in actions[1:]:
        same_action    = (action == prev_action)
        close_in_time  = (frame_idx - prev_frame) <= gap_frames

        if same_action and close_in_time:
            prev_frame, prev_ts = frame_idx, ts
        else:
            events.append({
                "action":      prev_action,
                "start_ts":    start_ts,
                "end_ts":      prev_ts,
                "start_frame": start_frame,
                "end_frame":   prev_frame,
            })
            start_frame, start_ts = frame_idx, ts
            prev_frame, prev_action, prev_ts = frame_idx, action, ts

    events.append({
        "action":      prev_action,
        "start_ts":    start_ts,
        "end_ts":      prev_ts,
        "start_frame": start_frame,
        "end_frame":   prev_frame,
    })
    return events


# ----------------------------
# MAIN INFERENCE
# ----------------------------
def run_pose_inference(input_path: str, output_path: str):
    actions = []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    labeled   = 0
    skipped   = 0

    print(f"Processing {total} frames at {fps:.1f} fps → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps if fps > 0 else 0.0

        # run pose estimation (detects people + keypoints in one pass)
        results = pose_model(
            frame,
            conf=CONF,
            iou=IOU,
            classes=[0],   # person only
            verbose=False
        )

        for r in results:
            boxes     = get_boxes_from_result(r)
            keypoints = extract_keypoints_from_result(r)

            if keypoints is None:
                # draw gray boxes for detected players with no usable keypoints
                for (x1, y1, x2, y2) in boxes:
                    w, h = x2 - x1, y2 - y1
                    if w < MIN_W or h < MIN_H:
                        continue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                skipped += 1
                continue

            action, conf_val = classify_pose(keypoints)

            # use largest box for the label (matches keypoint selection logic)
            if boxes:
                if len(boxes) > 1:
                    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
                    main_box = boxes[int(np.argmax(areas))]
                else:
                    main_box = boxes[0]

                x1, y1, x2, y2 = main_box
                w, h = x2 - x1, y2 - y1

                if w >= MIN_W and h >= MIN_H:
                    if action is not None:
                        actions.append((frame_idx, action, round(timestamp_sec, 2)))
                        draw_label(frame, x1, y1, x2, y2, action, conf_val, (0, 200, 80))
                        labeled += 1
                    else:
                        # below confidence threshold — show box + raw confidence
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        if conf_val is not None:
                            cv2.putText(
                                frame, f"{conf_val:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (150, 150, 150), 1
                            )
                        skipped += 1

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} — labeled: {labeled}, below threshold: {skipped}")

    events = deduplicate_actions(sorted(actions, key=lambda x: x[0]))

    cap.release()
    out.release()

    print(f"\n===== DONE =====")
    print(f"Frames processed : {frame_idx}")
    print(f"Labeled          : {labeled}")
    print(f"Below threshold  : {skipped}")
    print(f"Events detected  : {len(events)}")
    print(f"Output           : {output_path}")

    return events


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    events = run_pose_inference(INPUT_PATH, OUTPUT_PATH)

    print("\n--- Action Events ---")
    for e in events:
        print(f"  [{e['start_ts']}s → {e['end_ts']}s] {e['action'].upper()}"
              f"  (frames {e['start_frame']}–{e['end_frame']})")

    # save events to JSON alongside the video
    events_path = OUTPUT_PATH.replace(".mp4", "_events.json")
    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"\nEvents saved → {events_path}")
