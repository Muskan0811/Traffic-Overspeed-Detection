import os
import cv2

def extract_frames():
    video_dir = "input_videos"
    output_root = "extracted_frames"

    if os.path.exists(output_root) and len(os.listdir(output_root)) > 0:
        print("⏭ Skipping extraction (already done)")
        return

    os.makedirs(output_root, exist_ok=True)

    for video_name in os.listdir(video_dir):
        if not video_name.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        count = 0
        saved = 0

        video_id = video_name.split(".")[0]
        video_output_dir = os.path.join(output_root, video_id)
        os.makedirs(video_output_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % 5 == 0:
                frame_name = f"frame_{saved:05d}.jpg"
                cv2.imwrite(os.path.join(video_output_dir, frame_name), frame)
                saved += 1

            count += 1

        cap.release()

    print("✅ Frames extracted")