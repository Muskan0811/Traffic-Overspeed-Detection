import os
import cv2
import math
from ultralytics import YOLO

def track_and_speed():
    input_folder = "denoised_frames"
    output_folder = "final_frames"

    os.makedirs(output_folder, exist_ok=True)

    model = YOLO("yolov8n.pt")

    positions = {}
    prev_speeds = {}

    fps = 20
    SPEED_LIMIT = 60   # adjust later

    FRAME_SKIP = 2   # 🔥 process every 2nd frame (big speed boost)

    image_list = sorted(os.listdir(input_folder))

    for idx, img_name in enumerate(image_list):

        # 🔥 skip frames for speed
        if idx % FRAME_SKIP != 0:
            continue

        print(f"Processing frame {idx+1}/{len(image_list)}")

        path = os.path.join(input_folder, img_name)
        frame = cv2.imread(path)

        if frame is None:
            continue

        # 🔥 resize for faster YOLO
        frame = cv2.resize(frame, (640, 384))

        results = model.track(frame, persist=True, verbose=False)

        for r in results:
            for box in r.boxes:

                if box.id is None:
                    continue

                obj_id = int(box.id[0])
                cls = int(box.cls[0])

                if cls not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                speed = 0

                if obj_id in positions:
                    prev_x, prev_y = positions[obj_id]

                    dist = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    speed = dist * fps

                    # smoothing
                    if obj_id in prev_speeds:
                        speed = 0.7 * prev_speeds[obj_id] + 0.3 * speed

                positions[obj_id] = (cx, cy)
                prev_speeds[obj_id] = speed

                # 🚨 overspeed check
                if speed > SPEED_LIMIT:
                    color = (0, 0, 255)
                    label = f"ID {obj_id} | OVERSPEED!"
                else:
                    color = (0, 255, 0)
                    label = f"ID {obj_id} | {int(speed)} px/s"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                if speed > SPEED_LIMIT:
                    cv2.putText(frame, "OVERSPEEDING DETECTED",
                                (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

        cv2.imwrite(os.path.join(output_folder, img_name), frame)

    print("✅ Tracking + Speed + Overspeed done")