import os
import cv2
import math
import time
from ultralytics import YOLO


def track_and_speed():
    input_folder = "denoised_frames"
    output_folder = "final_frames"

    os.makedirs(output_folder, exist_ok=True)

    model = YOLO("yolov8n.pt")

    # 🔥 history storage
    positions = {}     # [(x,y), ...]
    timestamps = {}    # [t1, t2, ...]
    kalman = {}        # smoothed speed

    # 🔥 config
    SPEED_LIMIT = 15        # now in m/s (~54 km/h)
    FRAME_SKIP = 2
    WINDOW_SIZE = 6
    SCALE = 0.05           # meters per pixel (adjust based on camera)

    image_list = sorted(os.listdir(input_folder))

    for idx, img_name in enumerate(image_list):

        # skip frames
        if idx % FRAME_SKIP != 0:
            continue

        print(f"Processing frame {idx+1}/{len(image_list)}")

        path = os.path.join(input_folder, img_name)
        frame = cv2.imread(path)

        if frame is None:
            continue

        # resize
        frame = cv2.resize(frame, (640, 384))

        results = model.track(frame, persist=True, verbose=False)

        for r in results:
            for box in r.boxes:

                if box.id is None:
                    continue

                obj_id = int(box.id[0])
                cls = int(box.cls[0])

                # vehicle classes
                if cls not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_time = time.time()

                # initialize
                if obj_id not in positions:
                    positions[obj_id] = []
                    timestamps[obj_id] = []

                positions[obj_id].append((cx, cy))
                timestamps[obj_id].append(current_time)

                # keep sliding window
                if len(positions[obj_id]) > WINDOW_SIZE:
                    positions[obj_id].pop(0)
                    timestamps[obj_id].pop(0)

                speed = 0

                # 🔥 multi-frame speed (first → last)
                if len(positions[obj_id]) >= 2:

                    x_start, y_start = positions[obj_id][0]
                    x_end, y_end = positions[obj_id][-1]

                    t_start = timestamps[obj_id][0]
                    t_end = timestamps[obj_id][-1]

                    dist_pixels = math.sqrt(
                        (x_end - x_start) ** 2 + (y_end - y_start) ** 2
                    )

                    delta_t = t_end - t_start

                    if delta_t > 0:
                        # convert to meters/sec
                        speed = (dist_pixels * SCALE) / delta_t

                        # 🔥 Kalman-like smoothing
                        if obj_id not in kalman:
                            kalman[obj_id] = speed

                        kalman[obj_id] = 0.8 * kalman[obj_id] + 0.2 * speed
                        speed = kalman[obj_id]

                # convert to km/h
                speed_kmh = speed * 3.6

                # overspeed check
                if speed > SPEED_LIMIT:
                    color = (0, 0, 255)
                    label = f"ID {obj_id} | OVERSPEED {int(speed_kmh)} km/h"
                else:
                    color = (0, 255, 0)
                    label = f"ID {obj_id} | {int(speed_kmh)} km/h"

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # label
                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                # alert text
                if speed > SPEED_LIMIT:
                    cv2.putText(frame, "OVERSPEEDING DETECTED",
                                (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

        cv2.imwrite(os.path.join(output_folder, img_name), frame)

    print("✅ Tracking + Multi-frame Speed + Overspeed done")