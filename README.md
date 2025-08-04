# Face-Recognition-Attendance-Tracking-System-from-CCTV-SurveillanceThis project is a Real-Time AI Attendance System that detects, recognizes, and logs attendance for multiple employees using RTSP video feeds, face recognition (via ArcFace), person detection (YOLOv8), and a robust FAISS-based search backend for fast face matching.

🚀 Features
Multi-person tracking with YOLOv8 and ByteTrack

Face recognition using ArcFace embeddings and FAISS index

RTSP support for real-world camera integration

Line-crossing logic to detect actual entries

Attendance logging with deduplication (per session and per day)

CSV export for audit and debugging

Video output with detailed visual overlays

Professional visualization for bounding boxes, stats, faces, and entry tracking

🧠 System Architecture
text
Copy
Edit
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
🗂️ Project Structure
text
Copy
Edit
.
├── config.json                # Camera and system configuration
├── main.py                   # Main script (this file)
├── face_register.py          # Face/image upload and DB registration
├── utils/
│   ├── embedding_model.py    # ArcFace embedder
│   ├── detect_face.py        # Face detection helper
│   ├── db_handler.py         # Database interface
├── models/
│   ├── yolov8n.pt            # Person detection model
│   ├── yolov8m-face-lindevs.pt  # Face detection model
├── faiss_index.bin           # FAISS index (auto-generated)
├── employee_ids.pkl          # Employee ID mapping (auto-generated)
├── output_video_PROFESSIONAL.mp4 # Saved output video
├── entry_track_log_professional.csv # CSV log of tracking and attendance
🧾 Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/<your-username>/ai-attendance-system.git
cd ai-attendance-system
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
💡 Make sure ultralytics, faiss-cpu, torch, opencv-python, cvzone, and other listed dependencies are installed.

3. Configure Camera and System Settings
Update the config.json file with:

json
Copy
Edit
{
  "entry_video": {
    "path": "rtsp://your-camera-stream",
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
🖼️ Register Employees
Use face_register.py to register employee photos:

bash
Copy
Edit
python face_register.py
This script:

Uploads employee images

Stores metadata in the database

Images are embedded using ArcFace and saved for recognition

▶️ Run the System
bash
Copy
Edit
python main.py
System will:

Generate embeddings for any new employees

Build a FAISS index

Start video capture

Detect, track, and recognize faces

Log attendance into your database

Save annotated output video and CSV logs

✅ Attendance Logging Logic
Logs only once per session per person

Checks whether attendance was already marked today

Attendance triggered only after crossing a virtual line

Highly configurable and adjustable logic

📊 Output
output_video_PROFESSIONAL.mp4: Rendered video with bounding boxes, labels, and status

entry_track_log_professional.csv: Detailed log per frame with recognition status

Attendance records: Logged to your database

🧪 Testing
You can also test with a saved video feed by updating:

json
Copy
Edit
"path": "sample_video.mp4"
in your config.json.

🔧 Customization
Line crossing tolerance

Frame interval

FPS control

Retry logic

Track expiry and deduplication windows

Visualization styling

All can be tuned in main.py and config.json.

🧩 Dependencies
Ultralytics YOLOv8

ArcFace

FAISS

OpenCV, NumPy, cvzone, Torch, etc.
