import os
import cv2

def denoise_frames():
    input_folder = "all_frames"
    output_folder = "denoised_frames"

    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print("⏭ Skipping denoise (already done)")
        return

    os.makedirs(output_folder, exist_ok=True)

    for img_name in sorted(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv2.imwrite(os.path.join(output_folder, img_name), denoised)

    print("✅ Denoising done")