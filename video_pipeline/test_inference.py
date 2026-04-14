from ultralytics import YOLO
import cv2
import os
import random
import torch
from model.transformations import val_transform
import json

INPUT_PATH  = "videos/back_view.mp4"
OUTPUT_PATH = "output/back_view.mp4"
CONF        = 0.5
IOU         = 0.5
MIN_W       = 50
MIN_H       = 100

ACTION_CLASSES = ["spike", "set", "serve", "defense", "block"]

actions = []

player_model = YOLO("model/weights/player_model.pt")

from model.cnn import CNN
 
CONFIDENCE_THRESHOLD = 0.75
 
_device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
 
checkpoint = torch.load("model/weights/action_model.pt", map_location=_device)
_num_classes = checkpoint["network.19.weight"].shape[0]
_action_model = CNN(num_classes=_num_classes).to(_device)
 
# Initialize LazyLinear with a dummy forward pass before loading weights
# UPDATED: Matches the new transformations.py SIZE = (146, 32)
_dummy = torch.zeros(1, 3, 146, 32).to(_device)
_ = _action_model(_dummy)
 
_action_model.load_state_dict(checkpoint)
_action_model.eval()

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {INPUT_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

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

    cv2.putText(
        frame,
        label,
        (bg_x1 + padding, bg_y2 - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )

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
            prev_frame, prev_ts = frame_idx, ts  # extend window
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

def run_video_inference():

    frame_idx = 0
    print(f"Processing {total} frames at {fps:.1f} fps → {OUTPUT_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps if fps else 0.0

        #results = player_model(frame, conf=CONF, iou=IOU, verbose=False)
        results = player_model(frame, conf=CONF, iou=IOU, 
                       classes=[0],  # person class only
                       verbose=False)
        for r in results:
            boxes = r.boxes.xyxy.cpu()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1

                if w < MIN_W or h < MIN_H:
                    continue

                tensor = crop_player(frame, x1, y1, x2, y2)
                action = classify_action(tensor)

                if action is not None:
                    actions.append((frame_idx, action, round(timestamp_sec, 2)))

                if action is not None:
                    draw_label(frame, x1, y1, x2, y2, action.upper(), (0, 255, 0))
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        out.write(frame)
        frame_idx += 1

    events = deduplicate_actions(sorted(actions, key=lambda x: x[0]))

    cap.release()
    out.release()

    return events

def save_inference_samples(cap, player_model, num_samples=16, output_path="output/inference_samples2.png"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    samples = []  # (raw_crop_rgb, transformed_tensor, action_label, confidence)

    frame_indices = [int(i * total_frames / (num_samples * 3)) for i in range(num_samples * 3)]

    for fi in frame_indices:
        if len(samples) >= num_samples:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        results = player_model(frame, conf=CONF, iou=IOU, verbose=False)
        for r in results:
            if len(samples) >= num_samples:
                break
            for box in r.boxes.xyxy.cpu():
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w < MIN_W or h < MIN_H:
                    continue

                # Raw crop (what YOLO gives us)
                crop_bgr = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                # Transformed (what the CNN sees)
                tensor = val_transform(crop_rgb).unsqueeze(0)

                # Get prediction
                tensor_dev = tensor.to(_device)
                with torch.no_grad():
                    logits = _action_model(tensor_dev)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                label = ACTION_CLASSES[pred.item()]
                confidence = conf.item()

                samples.append((crop_rgb, tensor.squeeze(0), label, confidence))
                break  # one crop per frame

    # Plot
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 2.5))
    fig.suptitle("Left: Raw YOLO Crop | Right: After val_transform (CNN input)", fontsize=11, y=1.01)

    for i, (raw, transformed, label, conf) in enumerate(samples):
        # Raw crop - just resize for display
        # UPDATED: Matches the new 146x32 aspect ratio
        raw_resized = cv2.resize(raw, (32, 146))
        axes[i, 0].imshow(raw_resized)
        axes[i, 0].set_title(f"Raw crop\n{raw.shape[1]}x{raw.shape[0]}", fontsize=7)
        axes[i, 0].axis("off")

        # Transformed - undo normalization for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display = (transformed * std + mean).clamp(0, 1)
        display = display.permute(1, 2, 0).numpy()

        axes[i, 1].imshow(display)
        # UPDATED: Title string matches new dimensions
        axes[i, 1].set_title(f"CNN input (146x32)\n{label} {conf:.2f}", fontsize=7)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved inference samples → {output_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset cap for main inference

def analyze_crop_dimensions(player_model, max_crops=2000):
    import numpy as np

    cap_local = cv2.VideoCapture(INPUT_PATH)  # open fresh cap
    if not cap_local.isOpened():
        raise ValueError(f"Cannot open video: {INPUT_PATH}")

    widths, heights, aspect_ratios = [], [], []
    collected = 0

    while True:
        if collected >= max_crops:
            break

        ret, frame = cap_local.read()
        if not ret:
            break

        results = player_model(
            frame,
            conf=0.3,
            iou=IOU,
            classes=[0],
            verbose=False
        )

        for r in results:
            for box in r.boxes.xyxy.cpu():
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(h / w)
                collected += 1
                if collected >= max_crops:
                    break

    cap_local.release()  # close local cap when done

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    def stats(arr):
        if len(arr) == 0:
            return "EMPTY"
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    print("\n=== CROP DIMENSION ANALYSIS ===")
    print("samples:", len(widths))
    print("\nWIDTH:", stats(widths))
    print("\nHEIGHT:", stats(heights))
    print("\nASPECT RATIO (h/w):", stats(aspect_ratios))

    return {
        "width": stats(widths),
        "height": stats(heights),
        "aspect_ratio": stats(aspect_ratios),
    }

if __name__ == "__main__":
    save_inference_samples(cap, player_model, num_samples=20)
    events = run_video_inference()
    #print(events)
    #analyze_crop_dimensions(player_model)
