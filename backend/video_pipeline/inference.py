"""
This module contains the primary logic for transforming raw video output
into classified volleyball actions with timestamps.

Authors: Abiola Raji, Patrick Dang
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from video_pipeline.config import (
    DEVICE, YOLO_MODEL_PATH, POSE_CLASSIFIER_PATH,
    CONF, IOU, CONFIDENCE_THRESHOLD, MIN_W, MIN_H,
    IDX_TO_CLASS, GAP_FRAMES
)

# --- MODEL DEFINITION ---
class PoseClassifier(nn.Module):
    """
    4 layer MLP (MLP) designed for volleyball action classifaction from keypoints 
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(57, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 5)
        )
    def forward(self, x): return self.net(x)

# --- GLOBAL MODEL LOADING ---
#Load model to specified path from config. 
pose_model = YOLO(YOLO_MODEL_PATH)
try:
    pose_model.to(DEVICE)
except:
    pass

classifier = PoseClassifier().to(DEVICE)
classifier.load_state_dict(torch.load(POSE_CLASSIFIER_PATH, map_location=DEVICE))
classifier.eval()

# --- FEATURE ENGINEERING ---
def get_angle(a, b, c):
    """
    Calculates the inner angele between points a b and c. 

    Args:
        a, b, c (np.array): 2D coordinates representing the joints. 
                           'b' is the vertex of the angle.
    Returns:
        float: The angle in radians, clipped between 0 and Pi.

    """
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def normalize_keypoints(kp):
    """
    Transforms yolo keypoints into a normalized feature vector for classification

    Args:
        kp (np.array): Raw keypoint data from YOLO.
    Returns:
        np.array: A 57-dimensional feature vector for the Pose Classifier.
    """
    kp = np.array(kp)
    xy, conf = kp[:, :2].copy(), kp[:, 2:]
    xy[conf.squeeze() < 0.3] = 0
    center = (xy[11] + xy[12]) / 2
    xy -= center
    torso = np.linalg.norm((xy[5] + xy[6]) / 2)
    if torso > 0: xy /= torso
    if xy[6][0] < xy[5][0]: xy[:, 0] *= -1
    
    angles = [
        get_angle(xy[5], xy[7], xy[9]), get_angle(xy[6], xy[8], xy[10]),
        get_angle(xy[11], xy[13], xy[15]), get_angle(xy[12], xy[14], xy[16]),
        get_angle(xy[7], xy[5], xy[11]), get_angle(xy[8], xy[6], xy[12])
    ]
    return np.concatenate([xy.flatten(), conf.flatten(), np.array(angles)])

# --- PROCESSING HELPERS ---
def classify_pose(keypoints_raw):
    """
    Runs nomralized keypoints through classifier to get action label and confidence.

    Args:
        keypoints_raw (np.array): Raw keypoints for a single person detection.
    Returns:
        tuple: Returns (None, conf) if confidence is below the threshold.
    """
    try:
        features = normalize_keypoints(keypoints_raw)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        if conf.item() < CONFIDENCE_THRESHOLD:
            return None, conf.item()
        return IDX_TO_CLASS[pred.item()], conf.item()
    except:
        return None, None

def extract_keypoints_and_boxes(result):
    """
    Parses Yolo inference results to extract bounding boxes and keypoints

    if mulitple detections are presetent take the largest box.

    Args: result object from yolo

    Returns: tuple (keypoints for classification, bounding boxes)
    """
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None, []
    kp = result.keypoints.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    # Return kp for largest box and all boxes
    if len(kp) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return kp[areas.argmax()], boxes
    return kp[0], boxes

def draw_label(frame, box, label, conf, color=(0, 200, 80)):
    """
    Draws visual bounding boxes and action labels onto the video. 

    Args:
        frame (np.array): The current video frame.
        box (list/np.array): Bounding box coordinates [x1, y1, x2, y2].
        label (str): The predicted action name.
        conf (float): The confidence score of the prediction.
        color (tuple): colour for bounding box frame 

    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label: return
    text = f"{label.upper()} {conf:.2f}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(y1-h-10, 0)), (x1+w+10, y1), color, -1)
    cv2.putText(frame, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def deduplicate_actions(actions):
    """
    Groups frame level action cdetections into continious timed events. 

    Uses a frame threshold to merge like frames into one action

    Args:
        actions (list): A list of tuples (frame_idx, action_label, timestamp) sorted by frame_idx.
    Returns:
        list: A list of dictionaries with keys: action, start_ts, end_ts, start_frame, end_frame.
    """
    if not actions: return []
    events = []
    pf, pa, pts = actions[0]
    sf, sts = pf, pts
    for f, a, t in actions[1:]:
        if a == pa and (f - pf) <= GAP_FRAMES:
            pf, pts = f, t
        else:
            events.append({"action": pa, "start_ts": sts, "end_ts": pts, "start_frame": sf, "end_frame": pf})
            sf, sts, pf, pa, pts = f, t, f, a, t
    events.append({"action": pa, "start_ts": sts, "end_ts": pts, "start_frame": sf, "end_frame": pf})
    return events

def run_video_inference(input_path: str, output_path: str):
    """
    Main pipeline entry point called by app.py.
    Processes video, runs pose estimation, classifies actions, and saves output.
    """
    actions = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Using 'mp4v' or 'avc1' depending on your system's codec support
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = round(frame_idx / fps, 2) if fps > 0 else 0.0

        # 1. Run Pose Estimation
        results = pose_model(frame, conf=CONF, iou=IOU, classes=[0], verbose=False)

        for r in results:
            # 2. Extract keypoints for classification and boxes for drawing
            kp, boxes = extract_keypoints_and_boxes(r)
            
            if kp is not None:
                # 3. Classify the action based on pose
                action, conf_val = classify_pose(kp)
                
                # Determine the main box (largest) to attach the label to
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                main_box = boxes[int(np.argmax(areas))]
                bx1, by1, bx2, by2 = main_box

                # 4. Filter by size and confidence
                if (bx2-bx1) >= MIN_W and (by2-by1) >= MIN_H:
                    if action:
                        actions.append((frame_idx, action, timestamp_sec))
                        draw_label(frame, main_box, action, conf_val)
                    else:
                        # Draw a simple box for detected players without a high-conf action
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # 5. Group individual frame detections into timed events
    events = deduplicate_actions(sorted(actions, key=lambda x: x[0]))
    return events