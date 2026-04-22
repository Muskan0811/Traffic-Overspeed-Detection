import streamlit as st
import os
import shutil
import stat

from extract import extract_frames
from merge import merge_frames
from denoise import denoise_frames
from tracking import track_and_speed
from video import create_video

st.set_page_config(page_title="Traffic Analyzer", layout="centered")

st.title("🚗 Traffic Video Analyzer (Overspeed Detection)")

# ---------- helper: force delete (fix Mac error) ----------
def force_delete(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_delete(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, onerror=force_delete)

# ---------- upload ----------
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4"])

if uploaded_file is not None:

    input_folder = "input_videos"
    os.makedirs(input_folder, exist_ok=True)

    video_path = os.path.join(input_folder, uploaded_file.name)

    # overwrite old video (important)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Video uploaded successfully")

    if st.button("🚀 Run Detection"):

        # ---------- clean previous outputs ----------
        folders_to_clear = [
            "extracted_frames",
            "all_frames",
            "denoised_frames",
            "final_frames",
            "output_video"
        ]

        for folder in folders_to_clear:
            safe_delete(folder)

        # ---------- run pipeline ----------
        with st.spinner("Processing video... please wait ⏳"):
            try:
                extract_frames()
                merge_frames()
                denoise_frames()
                track_and_speed()
                create_video()
                st.success("🎉 Processing Complete!")

            except Exception as e:
                st.error(f"❌ Error occurred: {e}")

        # ---------- show output ----------
        output_video_path = "output_video/final_output.mp4"

        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)
        else:
            st.error("⚠️ Output video not found. Something went wrong.")