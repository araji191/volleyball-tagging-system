import os
import json
from tqdm import tqdm
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
INPUT_ROOT = "cropped_clean_split"
OUTPUT_ROOT = "keypoints_json3"

MODEL_NAME = "yolov8x-pose.pt"

IMG_SIZE = 512        # smaller = faster
BATCH_SIZE = 16       # adjust (8–32 depending on RAM)
CONF = 0.2            # lower = more detections

SPLITS = ["train", "valid", "test"]

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_NAME)

# Try Apple GPU (MPS)
try:
    model.to("mps")
    print("✅ Using MPS (Apple GPU)")
except:
    print("⚠️ Using CPU")

# ----------------------------
# HELPERS
# ----------------------------
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ----------------------------
# MAIN LOOP
# ----------------------------
total_processed = 0
total_skipped = 0

for split in SPLITS:
    split_input = os.path.join(INPUT_ROOT, split)
    split_output = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(split_output, exist_ok=True)

    for class_name in os.listdir(split_input):
        class_input = os.path.join(split_input, class_name)
        if not os.path.isdir(class_input):
            continue

        class_output = os.path.join(split_output, class_name)
        os.makedirs(class_output, exist_ok=True)

        image_files = [
            os.path.join(class_input, f)
            for f in os.listdir(class_input)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        # ----------------------------
        # BATCH PROCESSING
        # ----------------------------
        for batch in tqdm(list(chunk_list(image_files, BATCH_SIZE)),
                          desc=f"{split}/{class_name}"):

            results = model.predict(
                source=batch,
                imgsz=IMG_SIZE,
                conf=CONF,
                verbose=False
            )

            # ----------------------------
            # PROCESS EACH RESULT
            # ----------------------------
            for img_path, result in zip(batch, results):
                img_name = os.path.basename(img_path)

                try:
                    if result.keypoints is None:
                        total_skipped += 1
                        continue

                    kp_tensor = result.keypoints.data

                    if kp_tensor is None or len(kp_tensor) == 0:
                        total_skipped += 1
                        continue

                    keypoints = kp_tensor.cpu().numpy()

                    # pick main person
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

                    # convert to list
                    kp_list = keypoints.tolist()

                    data = {
                        "image": img_name,
                        "class": class_name,
                        "keypoints": kp_list
                    }

                    save_path = os.path.join(
                        class_output,
                        os.path.splitext(img_name)[0] + ".json"
                    )

                    with open(save_path, "w") as f:
                        json.dump(data, f)

                    total_processed += 1

                except Exception as e:
                    print(f"Error: {img_name} -> {e}")
                    total_skipped += 1

# ----------------------------
# SUMMARY
# ----------------------------
print("\n===== DONE =====")
print(f"Processed: {total_processed}")
print(f"Skipped:   {total_skipped}")
print(f"Success rate: {total_processed / (total_processed + total_skipped + 1e-6):.2%}")