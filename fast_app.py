import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import math
import os

st.title("🚗 Fast Traffic Analyzer (Overspeed Detection)")

uploaded_file = st.file_uploader("Upload video", type=["mp4"])

if uploaded_file is not None:

    # save uploaded file properly
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()  # 🔥 IMPORTANT

    st.success("✅ Uploaded")

    if st.button("🚀 Run Detection"):

        model = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(tfile.name)

        fps = 20
        positions = {}
        prev_speeds = {}
        SPEED_LIMIT = 60

        output_path = "output.mp4"

        # 🔥 FIXED codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out = None
        frame_count = 0

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 skip frames
            if frame_count % 3 != 0:
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

                    speed = 0

                    if obj_id in positions:
                        px, py = positions[obj_id]
                        dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                        speed = dist * fps

                        if obj_id in prev_speeds:
                            speed = 0.7 * prev_speeds[obj_id] + 0.3 * speed

                    positions[obj_id] = (cx, cy)
                    prev_speeds[obj_id] = speed

                    if speed > SPEED_LIMIT:
                        color = (0, 0, 255)
                        label = f"ID {obj_id} OVERSPEED"
                    else:
                        color = (0, 255, 0)
                        label = f"{int(speed)}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 🔥 initialize writer once
            if out is None:
                h, w, _ = frame.shape
                out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

            out.write(frame)

            # 🔥 live preview
            stframe.image(frame, channels="BGR")

        cap.release()

        if out is not None:
            out.release()

        st.success("🎉 Done!")

        # 🔥 FIX: stream video properly
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
        else:
            st.error("❌ Video not created")