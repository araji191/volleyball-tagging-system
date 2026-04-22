import cv2
import os
import json
import torch
from video_pipeline.config import CONF, IOU, MIN_W, MIN_H
from video_pipeline.inference import (
    pose_model, classify_pose, extract_keypoints_and_boxes, 
    draw_label, deduplicate_actions
)

INPUT_PATH  = "./videos/rally.mp4"
OUTPUT_PATH = "output/rally.mp4"

def run_video_inference():
    actions = []
    cap = cv2.VideoCapture(INPUT_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        ts = round(frame_idx / fps, 2)
        results = pose_model(frame, conf=CONF, iou=IOU, classes=[0], verbose=False)

        for r in results:
            kp, boxes = extract_keypoints_and_boxes(r)
            if kp is not None:
                action, conf_val = classify_pose(kp)
                # Filter for largest person box
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                main_box = boxes[int(torch.argmax(torch.tensor(areas)))]
                
                bx1, by1, bx2, by2 = main_box
                if (bx2-bx1) >= MIN_W and (by2-by1) >= MIN_H:
                    if action:
                        actions.append((frame_idx, action, ts))
                        draw_label(frame, main_box, action, conf_val)
                    else:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    events = deduplicate_actions(sorted(actions, key=lambda x: x[0]))
    with open(OUTPUT_PATH.replace(".mp4", "_events.json"), "w") as f:
        json.dump(events, f, indent=2)
    return events

if __name__ == "__main__":
    print("Starting Pose-Inference...")
    events = run_video_inference()
    print(f"Done. Detected {len(events)} events.")