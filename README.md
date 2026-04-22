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

1. Install dependencies:

pip install -r requirements.txt

2. Run the application:

streamlit run fast_app.py

## Output

- Green bounding box: normal vehicle  
- Red bounding box: overspeeding vehicle  

## Demo

<img width="1600" height="1201" alt="PHOTO-2026-04-22-22-08-11" src="https://github.com/user-attachments/assets/fe98507f-a6fe-47c3-9656-7818b647a983" />


<img width="1600" height="1231" alt="PHOTO-2026-04-22-22-08-28" src="https://github.com/user-attachments/assets/84e83738-d8c4-4167-ad3e-34d1c7c89168" />




## Project Structure

Traffic-Overspeed-Detection/
├── fast_app.py  
├── tracking.py  
├── video.py  
├── extract.py  
├── merge.py  
├── denoise.py  
├── detect.py  
├── main.py  
├── requirements.txt  
├── README.md  
└── .gitignore  

## Author

Muskan Behera
