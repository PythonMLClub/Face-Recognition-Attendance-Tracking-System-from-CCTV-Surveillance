# 🎥 Face Recognition Attendance Tracking System from CCTV Surveillance

This project is a **Real-Time AI Attendance System** that detects, recognizes, and logs attendance for multiple employees using RTSP video feeds, face recognition (via ArcFace), person detection (YOLOv8), and a robust FAISS-based search backend for fast face matching.

---

## 🚀 Features

- 👥 Multi-person tracking with YOLOv8 and ByteTrack  
- 🧠 Face recognition using ArcFace embeddings and FAISS index  
- 🌐 RTSP support for real-world camera integration  
- 📏 Line-crossing logic to detect actual entries  
- 🗓️ Attendance logging with deduplication (per session and per day)  
- 📁 CSV export for audit and debugging  
- 🎞️ Output video with professional visual overlays  
- 📊 Debug stats, entry detection, and live recognition info  

---

## 🧠 System Architecture

Video Feed (RTSP)
      ↓
Frame Capture (Threaded Queue)
      ↓
YOLOv8 Person & Face Detection
      ↓
Face Cropping → ArcFace Embedding → FAISS Search
      ↓
Entry Line Crossing Logic
      ↓
Attendance Logging (Database) + CSV Logging
      ↓
Visualization + Output Video Rendering


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

git clone [https://github.com/<your-username>/ai-attendance-system](https://github.com/PythonMLClub/Face-Recognition-Attendance-Tracking-System-from-CCTV-Surveillance).git
cd ai-attendance-system


---

## 🔁 Full System Flow

📡 **RTSP Feed**  
- Configured via `config.json` → `"path": "rtsp://..."` or a video file.  
- Captures frames from CCTV/IP camera.

⬇️  

🧵 **Frame Queue (Threaded)**  
- Managed via a background thread.  
- Queues frames for processing with FPS control and delay handling.

⬇️  

🧍‍♂️ **YOLOv8 Detection (Person + Face)**  
- Uses `yolov8n.pt` for person detection (primary).  
- Uses `yolov8m-face-lindevs.pt` for face detection (fallback).  
- Uses ByteTrack to track people across frames.  

⬇️  

🧠 **ArcFace Embedding → FAISS Match**  
- Crops faces from frame → embeds using ArcFace (512-d vector).  
- FAISS index (`faiss_index.bin`) is used to find the closest employee embedding.  
- Controlled by `similarity_threshold` in `config.json`.

⬇️  

🛑 **Line Crossing Logic**  
- Custom logic tracks `y` position of each person.  
- Crossing triggers if person moves from above to below the virtual line.  
- Detects classic, gradual, trend, and initial below-line crossings.  

⬇️  

📝 **Attendance Logging + CSV**  
- Logged only if:  
  - Person is recognized  
  - Person crosses the line  
  - Person has not been logged today or in current session  
- Stored in the database  
- Backup written to `entry_track_log_professional.csv`

⬇️  

🎥 **Video Rendering with Overlays**  
- Bounding boxes, employee names, status, track ID, debug stats  
- Saved to `output_video_PROFESSIONAL.mp4`

---


---

## 🔁 Full System Flow

📡 **RTSP Feed**  
- Configured via `config.json` → `"path": "rtsp://..."` or a video file.  
- Captures frames from CCTV/IP camera.

⬇️  

🧵 **Frame Queue (Threaded)**  
- Managed via a background thread.  
- Queues frames for processing with FPS control and delay handling.

⬇️  

🧍‍♂️ **YOLOv8 Detection (Person + Face)**  
- Uses `yolov8n.pt` for person detection (primary).  
- Uses `yolov8m-face-lindevs.pt` for face detection (fallback).  
- Uses ByteTrack to track people across frames.  

⬇️  

🧠 **ArcFace Embedding → FAISS Match**  
- Crops faces from frame → embeds using ArcFace (512-d vector).  
- FAISS index (`faiss_index.bin`) is used to find the closest employee embedding.  
- Controlled by `similarity_threshold` in `config.json`.

⬇️  

🛑 **Line Crossing Logic**  
- Custom logic tracks `y` position of each person.  
- Crossing triggers if person moves from above to below the virtual line.  
- Detects classic, gradual, trend, and initial below-line crossings.  

⬇️  

📝 **Attendance Logging + CSV**  
- Logged only if:  
  - Person is recognized  
  - Person crosses the line  
  - Person has not been logged today or in current session  
- Stored in the database  
- Backup written to `entry_track_log_professional.csv`

⬇️  

🎥 **Video Rendering with Overlays**  
- Bounding boxes, employee names, status, track ID, debug stats  
- Saved to `output_video_PROFESSIONAL.mp4`

---

## ⚙️ Configuration (via `config.json`)

{
  "entry_video": {
    "path": "rtsp://your-camera-url-or-video-file.mp4",
    "width": 1020,
    "height": 600,
    "fps": 5,
    "line_position": 0.4,
    "display_label": "ENTRY",
    "camera_id": "CAM1",
    "event_type": "Entry",
    "location": "Entry Gate",
    "output_path": "output_video_PROFESSIONAL.mp4",
    "max_retries": 5
  },
  "similarity_threshold": 1.5
}







