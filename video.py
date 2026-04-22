import os
import cv2

def create_video():
    image_folder = "final_frames"
    output_path = "output_video/final_output.mp4"

    os.makedirs("output_video", exist_ok=True)

    images = sorted(os.listdir(image_folder))

    if len(images) == 0:
        print("❌ No images found")
        return

    first_frame = cv2.imread(os.path.join(image_folder, images[0]))

    if first_frame is None:
        print("❌ First frame not readable")
        return

    height, width, _ = first_frame.shape

    # ✅ BEST codec for Mac + browser
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    video = cv2.VideoWriter(
        output_path,
        fourcc,
        20,
        (width, height)
    )

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

    print("🎥 Video created:", output_path)