from ultralytics import YOLO
import cv2
import os
import torch

# Paths and Config
INPUT_PATH  = "videos/rally.mp4"
OUTPUT_PATH = "output/rally.mp4"
CONF        = 0.6
IOU         = 0.5
MIN_W       = 50
MIN_H       = 100
CONFIDENCE_THRESHOLD = 0.9

# 1. Load both models
# player_model: Detection (finds the people)
# action_model: Classification (identifies the action)
player_model = YOLO("model/weights/player_model.pt")
action_model = YOLO("runs/classify/train/weights/best.pt") # Your new YOLO-cls model

actions = []

def classify_action_yolo(frame, x1, y1, x2, y2):
    # Crop the player from the BGR frame
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    # Run YOLO classification on the crop
    # YOLO handles the resize/transform automatically
    results = action_model.predict(crop, verbose=False)
    
    for r in results:
        conf = r.probs.top1conf.item()
        if conf >= CONFIDENCE_THRESHOLD:
            idx = r.probs.top1
            return r.names[idx] # Returns 'spike', 'set', etc.
            
    return None

def draw_label(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if not label: return
    
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thick)

def run_video_inference():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened(): raise ValueError("Video not found")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"Processing {total} frames...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        timestamp = round(frame_idx / fps, 2)
        
        # 1. Detect Players
        detection_results = player_model(frame, conf=CONF, iou=IOU, verbose=False)

        for r in detection_results:
            for box in r.boxes.xyxy.cpu():
                x1, y1, x2, y2 = map(int, box)
                
                if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                    continue

                # 2. Classify Action using the new YOLO model
                action = classify_action_yolo(frame, x1, y1, x2, y2)

                if action:
                    actions.append((action, timestamp))
                    draw_label(frame, x1, y1, x2, y2, action.upper(), (0, 255, 0))
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Inference Complete.")

if __name__ == "__main__":
    run_video_inference()
    # Deduplicate actions or print timeline
    print(actions)