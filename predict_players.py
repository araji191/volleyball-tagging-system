from ultralytics import YOLO
import cv2
import os

print("hi")

model = YOLO("model/weights/player_model.pt")
print("hi")

img_path = "samples/test_players.jpg"

img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found or cannot be read")

draw_img = img.copy()

os.makedirs("crops", exist_ok=True)

results = model(img_path, conf=0.3, iou=0.5)

print("hi")

crop_count = 0

for r in results:
    boxes = r.boxes.xyxy
    confs = r.boxes.conf

    for box, conf in zip(boxes, confs):
        if conf < 0.3:
            continue

        x1, y1, x2, y2 = map(int, box)

        w = x2 - x1
        h = y2 - y1

        if w < 50 or h < 100:
            continue

        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(draw_img, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        crop = img[y1:y2, x1:x2]

        cv2.imwrite(f"crops/player_{crop_count}.jpg", crop)
        crop_count += 1

print(f"Saved {crop_count} players")

cv2.imshow("Filtered Players", draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()