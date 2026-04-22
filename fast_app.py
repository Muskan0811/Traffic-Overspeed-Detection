import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import math
import os
import time

st.title("Traffic Overspeed Detection System")

uploaded_file = st.file_uploader("Upload video", type=["mp4"])

if uploaded_file is not None:

    # save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    st.success("Video uploaded")

    if st.button("Run Detection"):

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(tfile.name)

        # 🔥 tracking memory
        positions = {}
        timestamps = {}
        kalman = {}

        # 🔥 config
        SPEED_LIMIT = 10      # m/s (~54 km/h)
        WINDOW_SIZE = 6
        SCALE = 0.05
        FRAME_SKIP = 2

        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out = None
        frame_count = 0

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # skip frames
            if frame_count % FRAME_SKIP != 0:
                continue

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

                    current_time = time.time()

                    if obj_id not in positions:
                        positions[obj_id] = []
                        timestamps[obj_id] = []

                    positions[obj_id].append((cx, cy))
                    timestamps[obj_id].append(current_time)

                    if len(positions[obj_id]) > WINDOW_SIZE:
                        positions[obj_id].pop(0)
                        timestamps[obj_id].pop(0)

                    speed = 0

                    # 🔥 multi-frame speed
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
                            speed = (dist_pixels * SCALE) / delta_t

                            # smoothing
                            if obj_id not in kalman:
                                kalman[obj_id] = speed

                            kalman[obj_id] = 0.8 * kalman[obj_id] + 0.2 * speed
                            speed = kalman[obj_id]

                    speed_kmh = speed * 3.6

                    if speed > SPEED_LIMIT:
                        color = (0, 0, 255)
                        label = f"ID {obj_id} OVERSPEED {int(speed_kmh)} km/h"
                    else:
                        color = (0, 255, 0)
                        label = f"ID {obj_id} {int(speed_kmh)} km/h"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(frame, label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

            if out is None:
                h, w, _ = frame.shape
                out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

            out.write(frame)

            stframe.image(frame, channels="BGR")

        cap.release()

        if out is not None:
            out.release()

        st.success("Processing complete")

        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
        else:
            st.error("Video not created")