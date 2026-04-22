import os
import shutil

def merge_frames():
    input_root = "extracted_frames"
    output_folder = "all_frames"

    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print("⏭ Skipping merge (already done)")
        return

    os.makedirs(output_folder, exist_ok=True)

    count = 0

    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)

        if not os.path.isdir(folder_path):
            continue

        for img in sorted(os.listdir(folder_path)):
            src = os.path.join(folder_path, img)
            dst = os.path.join(output_folder, f"frame_{count:06d}.jpg")

            shutil.copy(src, dst)
            count += 1

    print("✅ Frames merged")