import os
import cv2
from ultralytics import YOLO

def detect_vehicles():
    input_folder = "denoised_frames"
    output_folder = "detected_frames"

    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print("⏭ Skipping detection (already done)")
        return

    os.makedirs(output_folder, exist_ok=True)

    model = YOLO("yolov8n.pt")
    vehicle_classes = [2, 3, 5, 7]

    for img_name in sorted(os.listdir(input_folder)):
        path = os.path.join(input_folder, img_name)
        img = cv2.imread(path)

        if img is None:
            continue

        results = model(img)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_folder, img_name), img)

    print("✅ Detection done")