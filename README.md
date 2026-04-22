# Traffic Overspeed Detection System

An AI-based computer vision system that detects vehicles, tracks them across frames, estimates their speed, and identifies overspeeding vehicles from video input.

## Features

- Vehicle detection using YOLOv8
- Object tracking with unique IDs
- Speed estimation (pixel-based)
- Overspeed detection
- Streamlit web application interface

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Streamlit

## How to Run

```bash
pip install -r requirements.txt
streamlit run fast_app.py
