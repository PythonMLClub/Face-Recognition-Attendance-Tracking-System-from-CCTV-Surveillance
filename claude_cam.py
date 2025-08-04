import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import csv
import threading
import queue
import time
import json
import os
import faiss
import pickle
import sys

# Performance optimizations
cv2.setNumThreads(4)
cv2.setUseOptimized(True)
import torch
torch.set_num_threads(4)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from datetime import datetime
from utils.embedding_model import ArcFaceEmbedder
from utils.db_handler import get_employee_details, get_employees_without_embedding, update_embedding, get_all_embeddings, log_attendance
from utils.detect_face import detect_faces

# Adjust sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load config
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        video_config = config['entry_video']
        RTSP_URL = video_config['path']
        WIDTH = video_config.get('width', 1020)
        HEIGHT = video_config.get('height', 600)
        
        # FIXED: Much slower FPS for visible output
        FPS = 5  # CHANGED from 15 to 5 for much slower video
        
        # FIXED: Process every frame for complete visibility
        FRAME_INTERVAL = 1  # CHANGED from 5 to 1 - process every frame
        
        OUTPUT_PATH = video_config.get('output_path', 'output_video_PROFESSIONAL.mp4')  # Professional name
        
        # Handle line position
        line_position_value = video_config.get('line_position', 0.4)
        
        if line_position_value > 1:
            ACTUAL_LINE_Y = int(line_position_value)
            LINE_Y_RATIO = line_position_value / HEIGHT
            print(f"✅ Using absolute line position: {ACTUAL_LINE_Y} pixels")
        else:
            LINE_Y_RATIO = line_position_value
            ACTUAL_LINE_Y = int(HEIGHT * LINE_Y_RATIO)
            print(f"✅ Using ratio line position: {LINE_Y_RATIO} ({ACTUAL_LINE_Y} pixels)")
        
        DISPLAY_LABEL = video_config.get('display_label', 'ENTRY')
        CAMERA_ID = video_config.get('camera_id', 'CAM1')
        EVENT_TYPE = video_config.get('event_type', 'Entry')
        LOCATION = video_config.get('location', 'Entry Gate')
        MAX_RETRIES = video_config.get('max_retries', 5)
        SIMILARITY_THRESHOLD = config.get('similarity_threshold', 1.5)
        
        print(f"✅ Config loaded with PROFESSIONAL FPS: {FPS}")
        
except FileNotFoundError:
    print("❌ Error: config.json file not found.")
    exit(1)
except KeyError as e:
    print(f"❌ Error: Missing key in config.json: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error reading config.json: {e}")
    exit(1)

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')
face_model = YOLO('models/yolov8m-face-lindevs.pt')  # For face detection fallback

# Initialize ArcFace embedder
embedder = ArcFaceEmbedder()

# Disable OpenMP in FAISS
faiss.omp_set_num_threads(1)

# Track previous center positions
track_history = {}
in_count = 0
frame_queue = queue.Queue(maxsize=60)  # Increased queue size
stop_thread = False

# Global variables for better attendance tracking
employee_persistence = {}  # Initialize the global dictionary
attendance_logged_this_session = set()  # Track logged employees in this session
unique_id_counter = 50000  # Counter for truly unique IDs

DEDUPLICATION_WINDOW_SECONDS = 60
TRACK_EXPIRY_FRAMES = 45  # Reduced for better multi-person handling

# Line crossing detection parameters - OPTIMIZED FOR MULTIPLE PEOPLE
CROSSING_TOLERANCE = 8  # Reduced from 15 for better multi-person detection
MIN_HISTORY_FOR_CROSSING = 2  # Minimum positions needed for crossing detection

def get_unique_track_id():
    """Generate truly unique track ID"""
    global unique_id_counter
    unique_id_counter += 1
    return unique_id_counter

def detect_line_crossing_multiple(track_history, track_id, actual_line_y, frame_count):
    """ULTIMATE line crossing detection with multiple scenarios handled"""
    
    if track_id not in track_history:
        return False, None, None
        
    if track_history[track_id]['crossed']:
        return track_history[track_id]['crossed'], None, None
    
    history = track_history[track_id]['history']
    
    if len(history) < 1:  # Need at least one position
        return False, None, None
    
    curr_y = history[-1]
    prev_y = history[-2] if len(history) >= 2 else None
    
    crossed = False
    crossing_method = None
    
    # SOLUTION 1: Handle people detected BELOW the line initially
    if len(history) == 1 and curr_y > actual_line_y + 10:
        # Person first detected well below the line - assume they crossed
        crossed = True
        crossing_method = "initial_below_line"
        print(f"🎯 INITIAL BELOW LINE: TrackID {track_id} - FirstY:{curr_y}, LineY:{actual_line_y}")
    
    # SOLUTION 2: Classic crossing detection (above to below)
    if not crossed and prev_y is not None:
        if prev_y < actual_line_y and curr_y > actual_line_y:
            crossed = True
            crossing_method = "classic"
            print(f"🎯 CLASSIC CROSSING: TrackID {track_id} - PrevY:{prev_y}, CurrY:{curr_y}")
    
    # SOLUTION 3: Direct crossing with tolerance
    if not crossed and prev_y is not None:
        if (prev_y < actual_line_y - 3 and curr_y > actual_line_y + 3):
            crossed = True
            crossing_method = "direct_with_tolerance"
            print(f"🎯 DIRECT CROSSING: TrackID {track_id} - PrevY:{prev_y}, CurrY:{curr_y}")
    
    # SOLUTION 4: Gradual crossing (for slow movement)
    if not crossed and len(history) >= 3:
        # Check if person gradually moved from above to below line
        positions_above = sum(1 for y in history if y < actual_line_y)
        positions_below = sum(1 for y in history if y > actual_line_y)
        
        if positions_above >= 1 and positions_below >= 2 and curr_y > actual_line_y:
            crossed = True
            crossing_method = "gradual"
            print(f"🎯 GRADUAL CROSSING: TrackID {track_id} - Above:{positions_above}, Below:{positions_below}")
    
    # SOLUTION 5: Zone-based crossing (near line detection)
    if not crossed and prev_y is not None:
        line_zone_size = 15  # pixels around the line
        prev_in_zone = abs(prev_y - actual_line_y) <= line_zone_size
        curr_below_line = curr_y > actual_line_y + 5
        
        if prev_in_zone and curr_below_line:
            crossed = True
            crossing_method = "zone_based"
            print(f"🎯 ZONE CROSSING: TrackID {track_id} - PrevY:{prev_y}, CurrY:{curr_y}")
    
    # SOLUTION 6: Trend-based crossing (movement direction)
    if not crossed and len(history) >= 4:
        recent_movement = curr_y - history[-4]  # Movement over last 4 frames
        avg_position = sum(history[-4:]) / 4
        
        if (recent_movement > 8 and  # Moving downward
            avg_position <= actual_line_y + 10 and  # Near or past line
            curr_y > actual_line_y):  # Currently below line
            crossed = True
            crossing_method = "trend_based"
            print(f"🎯 TREND CROSSING: TrackID {track_id} - Movement:{recent_movement}, AvgPos:{avg_position}")
    
    # SOLUTION 7: Pattern-based crossing (scan entire history)
    if not crossed and len(history) >= 3:
        for i in range(len(history) - 1):
            y1 = history[i]
            y2 = history[i + 1]
            if y1 < actual_line_y and y2 > actual_line_y:
                crossed = True
                crossing_method = "pattern_scan"
                print(f"🎯 PATTERN CROSSING: TrackID {track_id} - Found crossing at positions {i}-{i+1}")
                break
    
    # SOLUTION 8: Proximity-based (been near line, now clearly past)
    if not crossed and len(history) >= 2:
        min_distance = min(abs(y - actual_line_y) for y in history)
        if min_distance <= 12 and curr_y > actual_line_y + 8:
            crossed = True
            crossing_method = "proximity"
            print(f"🎯 PROXIMITY CROSSING: TrackID {track_id} - MinDist:{min_distance}, CurrY:{curr_y}")
    
    # SOLUTION 9: Time-based crossing (person visible for several frames below line)
    if not crossed and len(history) >= 5:
        recent_below_count = sum(1 for y in history[-5:] if y > actual_line_y)
        if recent_below_count >= 4:  # 4 out of last 5 frames below line
            crossed = True
            crossing_method = "time_based"
            print(f"🎯 TIME-BASED CROSSING: TrackID {track_id} - {recent_below_count}/5 frames below line")
    
    # Enhanced debug info
    if frame_count % 5 == 0 and len(history) > 0:  # More frequent debugging
        distance = abs(curr_y - actual_line_y)
        position = "ABOVE" if curr_y < actual_line_y else "BELOW"
        print(f"🔍 TrackID {track_id} Frame {frame_count}: Y={curr_y} ({position} line), Dist={distance}, History={history[-3:]}")
    
    return crossed, prev_y, curr_y

def auto_adjust_line_position(track_history, current_line_y, frame_height):
    """Automatically suggest better line position based on tracking data"""
    
    if not track_history:
        return current_line_y
    
    # Collect all person positions
    all_positions = []
    for track_data in track_history.values():
        if track_data.get('face_matched', False):  # Only recognized people
            all_positions.extend(track_data.get('history', []))
    
    if len(all_positions) < 10:  # Not enough data
        return current_line_y
    
    # Calculate statistics
    avg_y = sum(all_positions) / len(all_positions)
    min_y = min(all_positions)
    max_y = max(all_positions)
    
    print(f"📊 POSITION ANALYSIS:")
    print(f"   Current line: {current_line_y}")
    print(f"   People positions - Min: {min_y}, Max: {max_y}, Avg: {avg_y}")
    
    # Suggest optimal line position (between min and average)
    optimal_line_y = int((min_y + avg_y) / 2)
    
    if abs(optimal_line_y - current_line_y) > 30:
        print(f"💡 SUGGESTION: Consider moving line to Y={optimal_line_y} for better detection")
        print(f"   This would be {(optimal_line_y/frame_height)*100:.1f}% from top")
    
    return current_line_y

def verify_line_position(frame_height, line_y, line_ratio):
    """Verify and adjust line position if needed"""
    
    print(f"📏 LINE POSITION VERIFICATION:")
    print(f"   Frame Height: {frame_height}")
    print(f"   Line Y: {line_y}")
    print(f"   Line Ratio: {line_ratio}")
    print(f"   Line Position: {(line_y/frame_height)*100:.1f}% from top")
    
    # Check if line position makes sense for entry detection
    if line_y < frame_height * 0.2:  # Too high (less than 20% from top)
        print("⚠️ WARNING: Line position seems too high for entry detection")
        suggested_y = int(frame_height * 0.4)
        print(f"   Suggested Y position: {suggested_y} (40% from top)")
    
    elif line_y > frame_height * 0.8:  # Too low (more than 80% from top)
        print("⚠️ WARNING: Line position seems too low for entry detection")
        suggested_y = int(frame_height * 0.6)
        print(f"   Suggested Y position: {suggested_y} (60% from top)")
    
    else:
        print("✅ Line position looks good for entry detection")
    
    return line_y

def manage_crowded_tracks(track_history, frame_count):
    """Better track management for crowded scenes"""
    
    # Identify tracks to clean up
    expired_tracks = []
    unmatched_old_tracks = []
    
    for track_id, data in track_history.items():
        frames_since_seen = frame_count - data['last_seen_frame']
        
        if frames_since_seen > TRACK_EXPIRY_FRAMES:
            expired_tracks.append(track_id)
        elif frames_since_seen > 20 and not data.get('face_matched', False):
            unmatched_old_tracks.append((track_id, frames_since_seen))
    
    # Remove expired tracks
    for track_id in expired_tracks:
        del track_history[track_id]
        print(f"🗑️ Removed expired TrackID {track_id}")
    
    # If too many tracks, remove oldest unmatched ones
    if len(track_history) > 20:  # Limit total tracks
        unmatched_old_tracks.sort(key=lambda x: x[1], reverse=True)
        for track_id, _ in unmatched_old_tracks[:5]:  # Remove oldest 5
            if track_id in track_history:
                del track_history[track_id]
                print(f"🗑️ Removed crowded TrackID {track_id}")

def enhanced_person_detection(frame, model, face_model, frame_count):
    """Enhanced person detection with unique track ID management"""
    
    # Primary detection with optimized parameters
    person_results = model.track(
        frame, 
        persist=True, 
        classes=[0],  # Person class
        conf=0.25,    # LOWERED from 0.5 - detects more people
        iou=0.5,      # NMS IoU threshold - prevents over-suppression
        max_det=15,   # Allow more detections per frame
        tracker="bytetrack.yaml"  # Use ByteTrack for better multi-object tracking
    )
    
    boxes = []
    person_ids = []
    confidences = []
    
    # Get person detections
    if person_results[0].boxes.id is not None:
        person_ids = person_results[0].boxes.id.cpu().numpy().astype(int).tolist()
        boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
        confidences = person_results[0].boxes.conf.cpu().numpy().tolist()
        print(f"Frame {frame_count}: Person Detection - Found {len(boxes)} people with IDs: {person_ids}")
    
    # Enhanced fallback with unique IDs
    if len(boxes) < 3:  # If fewer than 3 people detected, try face detection
        face_results = face_model(frame, conf=0.15, iou=0.4, max_det=10)
        
        if face_results[0].boxes is not None and len(face_results[0].boxes) > 0:
            face_boxes = face_results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            face_confidences = face_results[0].boxes.conf.cpu().numpy().tolist()
            
            # Convert face boxes to person-sized boxes (expand face to estimate person)
            expanded_boxes = []
            for i, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
                # Expand face box to estimate person box
                face_width = fx2 - fx1
                face_height = fy2 - fy1
                
                # Estimate person box (face is typically 1/8 of person height)
                person_width = int(face_width * 2.5)
                person_height = int(face_height * 6)
                
                # Calculate person box centered on face
                center_x = (fx1 + fx2) // 2
                center_y = (fy1 + fy2) // 2
                
                px1 = max(0, center_x - person_width // 2)
                py1 = max(0, center_y - person_height // 3)  # Face in upper part
                px2 = min(frame.shape[1], center_x + person_width // 2)
                py2 = min(frame.shape[0], center_y + (2 * person_height) // 3)
                
                expanded_boxes.append([px1, py1, px2, py2])
            
            # Add face-derived detections that don't overlap with existing person detections
            for i, face_box in enumerate(expanded_boxes):
                fx1, fy1, fx2, fy2 = face_box
                
                # Check for overlap with existing person detections
                overlap = False
                for px1, py1, px2, py2 in boxes:
                    # Calculate IoU
                    x1 = max(fx1, px1)
                    y1 = max(fy1, py1)
                    x2 = min(fx2, px2)
                    y2 = min(fy2, py2)
                    
                    if x1 < x2 and y1 < y2:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (fx2 - fx1) * (fy2 - fy1)
                        area2 = (px2 - px1) * (py2 - py1)
                        union = area1 + area2 - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.3:  # Significant overlap
                            overlap = True
                            break
                
                # Add non-overlapping face-derived detection with UNIQUE ID
                if not overlap:
                    boxes.append(face_box)
                    new_id = get_unique_track_id()
                    person_ids.append(new_id)
                    confidences.append(face_confidences[i] * 0.8)  # Lower confidence
                    print(f"Added face-derived person: ID {new_id} at {face_box}")
    
    # ADDITIONAL FALLBACK: Lower confidence person detection
    if len(boxes) < 2:  # If still not enough detections
        low_conf_results = model(frame, conf=0.15, classes=[0], max_det=15)
        if low_conf_results[0].boxes is not None and len(low_conf_results[0].boxes) > 0:
            low_conf_boxes = low_conf_results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            low_conf_confidences = low_conf_results[0].boxes.conf.cpu().numpy().tolist()
            
            # Filter out boxes that are too small (likely false positives)
            for i, box in enumerate(low_conf_boxes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                if width > 30 and height > 60:  # Minimum person size
                    # Check for overlap with existing detections
                    overlap = False
                    for existing_box in boxes:
                        ex1, ey1, ex2, ey2 = existing_box
                        if (abs(x1 - ex1) < 50 and abs(y1 - ey1) < 50):
                            overlap = True
                            break
                    
                    if not overlap:
                        boxes.append(box)
                        new_id = get_unique_track_id()
                        person_ids.append(new_id)
                        confidences.append(low_conf_confidences[i] * 0.6)
                        print(f"Added low-conf person: ID {new_id}")
    
    # Ensure we have confidence scores for all detections
    while len(confidences) < len(boxes):
        confidences.append(0.5)  # Default confidence
    
    return boxes, person_ids, confidences

def add_debug_visualization_professional(frame, track_history, actual_line_y, frame_count):
    """PROFESSIONAL debug visualization with properly sized elements."""
    
    # PROFESSIONAL: Entry line - thicker but not overwhelming
    cv2.line(frame, (0, actual_line_y), (frame.shape[1], actual_line_y), (0, 255, 255), 4)
    
    # PROFESSIONAL: Subtle line indicators
    cv2.line(frame, (0, actual_line_y-2), (frame.shape[1], actual_line_y-2), (0, 200, 200), 1)
    cv2.line(frame, (0, actual_line_y+2), (frame.shape[1], actual_line_y+2), (0, 200, 200), 1)
    
    # PROFESSIONAL: Smaller line label positioned on the right
    line_label = f"Entry Line Y={actual_line_y}"
    label_size = cv2.getTextSize(line_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (frame.shape[1] - label_size[0] - 20, actual_line_y - 30), 
                 (frame.shape[1] - 5, actual_line_y - 5), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, line_label, (frame.shape[1] - label_size[0] - 15, actual_line_y - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Calculate stats
    active_tracks = sum(1 for data in track_history.values() if len(data['history']) > 0)
    recognized_count = sum(1 for data in track_history.values() if data['face_matched'])
    unknown_count = active_tracks - recognized_count
    
    # PROFESSIONAL: Much smaller, compact stats box
    stats_bg_width = 280  # Reduced from 500
    stats_bg_height = 120  # Reduced from 220
    
    # Position in top-left with some margin
    cv2.rectangle(frame, (10, 10), (10 + stats_bg_width, 10 + stats_bg_height), 
                 (0, 0, 0), cv2.FILLED)  # Black background
    cv2.rectangle(frame, (10, 10), (10 + stats_bg_width, 10 + stats_bg_height), 
                 (100, 100, 100), 2)  # Gray border instead of white
    
    # PROFESSIONAL: Properly sized text
    text_size = 0.5  # Much smaller text
    text_thickness = 1  # Thinner text
    
    # Column 1 - Left side
    cv2.putText(frame, f"Frame: {frame_count}", (15, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness)
    cv2.putText(frame, f"Tracks: {active_tracks}", (15, 48), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness)
    cv2.putText(frame, f"Recognized: {recognized_count}", (15, 66), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), text_thickness)
    cv2.putText(frame, f"Unknown: {unknown_count}", (15, 84), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 100, 255), text_thickness)
    cv2.putText(frame, f"FPS: {FPS}", (15, 102), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 0), text_thickness)
    
    # Column 2 - Right side
    cv2.putText(frame, f"Entries: {in_count}", (150, 48), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 0), text_thickness)
    cv2.putText(frame, f"Attendance: {len(attendance_logged_this_session)}", (150, 66), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), text_thickness)
    cv2.putText(frame, "AI SYSTEM", (150, 84), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness)
    cv2.putText(frame, "ACTIVE", (150, 102), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), text_thickness)

def draw_professional_person_detection(display_frame, track_id, track_history, x1, y1, x2, y2):
    """Draw professional person detection with appropriately sized elements."""
    
    # Determine colors and status
    if track_history[track_id]['face_matched'] and track_history[track_id]['name']:
        # Green theme for recognized
        person_color = (0, 200, 0)
        text_color = (255, 255, 255)
        bg_color = (0, 150, 0)
        name = track_history[track_id]['name']
        emp_id = track_history[track_id]['employee_id']
        label = f"{name} (ID:{emp_id})"
    else:
        # Blue theme for unknown (less harsh than red)
        person_color = (255, 100, 0)  # Orange-blue
        text_color = (255, 255, 255)
        bg_color = (200, 80, 0)
        label = f"Track {track_id}"
    
    # PROFESSIONAL: Person bounding box - not too thick
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), person_color, 3)
    
    # PROFESSIONAL: Compact label
    text_size = 0.5  # Smaller text
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)[0]
    
    # PROFESSIONAL: Smaller label background
    label_bg_height = 20
    cv2.rectangle(display_frame, (x1, y1 - label_bg_height), 
                 (x1 + label_size[0] + 10, y1), bg_color, cv2.FILLED)
    
    # PROFESSIONAL: Smaller label text
    cv2.putText(display_frame, label, (x1 + 5, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)
    
    # PROFESSIONAL: Compact status indicator
    if track_history[track_id]['crossed']:
        status_text = "ENTERED"
        status_color = (0, 255, 255)  # Yellow
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(display_frame, (x1, y2 + 5), 
                     (x1 + status_size[0] + 10, y2 + 25), (0, 0, 0), cv2.FILLED)
        cv2.putText(display_frame, status_text, (x1 + 5, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

def draw_professional_face_detection(display_frame, face_x1, face_y1, face_x2, face_y2, 
                                   face_status, name=None):
    """Draw professional face detection boxes."""
    
    if face_status == "matched":
        # Green for matched
        face_color = (0, 255, 0)
        text = f"✓ {name}" if name else "✓ MATCHED"
        text_color = (0, 255, 0)
    elif face_status == "detected":
        # Yellow for detected but not matched
        face_color = (0, 255, 255)
        text = "FACE"
        text_color = (0, 255, 255)
    else:  # unknown
        # Orange for unknown
        face_color = (0, 165, 255)
        text = "UNKNOWN"
        text_color = (0, 165, 255)
    
    # PROFESSIONAL: Face box - not too thick
    cv2.rectangle(display_frame, (face_x1, face_y1), (face_x2, face_y2), face_color, 2)
    
    # PROFESSIONAL: Small face label
    text_size = 0.4
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)[0]
    
    # Position label above face box
    label_y = face_y1 - 5
    if label_y < 15:  # If too close to top, put below
        label_y = face_y2 + 15
    
    cv2.rectangle(display_frame, (face_x1, label_y - 12), 
                 (face_x1 + label_size[0] + 6, label_y + 2), (0, 0, 0), cv2.FILLED)
    cv2.putText(display_frame, text, (face_x1 + 3, label_y - 2), 
               cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)

# Generate embeddings for employees
def generate_embeddings():
    records = get_employees_without_embedding()
    if not records:
        print("No employees without embeddings found.")
        return {}

    employee_embeddings = {}
    for emp_id, image_blob in records:
        try:
            image = np.frombuffer(image_blob, dtype=np.uint8)
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ Failed to decode image for Employee ID: {emp_id}")
                continue

            faces = detect_faces(img)
            if not faces:
                print(f"❌ No face found for Employee ID: {emp_id}")
                continue

            face_crop, coords = faces[0]
            embedding = embedder.get_embedding(face_crop)
            if embedding is None:
                print(f"❌ Failed to generate embedding for Employee ID: {emp_id}")
                continue

            employee_embeddings[emp_id] = embedding
            success, message = update_embedding(emp_id, embedding)
            print(f"{'✅' if success else '❌'} {message} for Employee ID: {emp_id}")
        except Exception as e:
            print(f"❌ Error processing Employee ID {emp_id}: {e}")

    print("✅ Embeddings generation completed.")
    return employee_embeddings

# Build FAISS index
def build_faiss_index():
    try:
        records = get_all_embeddings()
        if not records:
            print("❌ No embeddings found in the database.")
            return None, None

        embeddings = []
        employee_ids = []
        for emp_id, embedding_blob in records:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            if embedding.shape[0] == 512:
                embeddings.append(embedding)
                employee_ids.append(str(emp_id))
            else:
                print(f"❌ Invalid embedding for Employee ID: {emp_id}")

        if not embeddings:
            print("❌ No valid embeddings to build FAISS index.")
            return None, None

        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(512)
        index.add(embeddings)

        faiss.write_index(index, "faiss_index.bin")
        with open("employee_ids.pkl", "wb") as f:
            pickle.dump(employee_ids, f)
        print("✅ FAISS index and employee IDs saved.")
        return index, employee_ids
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        return None, None

def capture_frames(rtsp_url):
    """Frame capture with better control for slow video."""
    retry_count = 0
    while not stop_thread:
        cap = cv2.VideoCapture(rtsp_url)
        
        # Better buffer settings for slower processing
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Limit input FPS
        
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream. Retrying ({retry_count+1}/{MAX_RETRIES})...")
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print("❌ Max retries reached. Exiting capture thread.")
                break
            time.sleep(2)
            continue
            
        retry_count = 0
        frame_capture_count = 0
        
        while not stop_thread:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame. Reconnecting...")
                break
                
            frame_capture_count += 1
            
            # Better frame queue management for slow processing
            try:
                # Keep only 2 frames max for slow processing
                while frame_queue.qsize() >= 2:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                frame_queue.put_nowait(frame)
                
                if frame_capture_count % 50 == 0:  # Log every 50 frames
                    print(f"📹 Captured {frame_capture_count} frames for PROFESSIONAL processing")
                    
            except queue.Full:
                pass  # Skip frame if queue is full
                
            # Slower capture rate
            time.sleep(0.05)  # 20 FPS capture rate
            
        cap.release()

def check_attendance_exists_today(employee_id):
    """Check if employee already has attendance logged today"""
    
    from datetime import datetime
    
    try:
        # This is a placeholder - implement based on your database
        # For now, we'll use the session tracking
        today_key = f"{employee_id}_{datetime.now().strftime('%Y-%m-%d')}"
        return employee_id in attendance_logged_this_session
        
    except Exception as e:
        print(f"❌ Error checking existing attendance: {e}")
        return False  # If error, proceed with logging

def log_employee_attendance(track_id, track_history, frame_count):
    """Improved attendance logging with better duplicate prevention"""
    
    global attendance_logged_this_session
    
    attendance_logged = track_history[track_id]['attendance_logged']
    
    if (track_history[track_id]['crossed'] and 
        track_history[track_id]['face_matched'] and 
        track_history[track_id]['employee_id'] and
        not attendance_logged):
        
        current_time = datetime.now()
        employee_id = track_history[track_id]['employee_id']
        employee_name = track_history[track_id]['name']
        
        print(f"🎯 CHECKING ATTENDANCE for Employee {employee_id} ({employee_name})")
        
        # Check if this employee has already been logged in this session
        if employee_id in attendance_logged_this_session:
            print(f"⚠️ Employee {employee_id} ({employee_name}) already logged in this session. Skipping.")
            track_history[track_id]['attendance_logged'] = True
            return False, "Already logged this session"
        
        # Additional check for today's attendance
        if check_attendance_exists_today(employee_id):
            print(f"⚠️ Employee {employee_id} already has attendance today. Skipping.")
            track_history[track_id]['attendance_logged'] = True
            attendance_logged_this_session.add(employee_id)
            return False, "Already logged today"
        
        # Log attendance
        timestamp = current_time
        confidence_score = track_history[track_id]['detection_confidence']
        
        success, message = log_attendance(employee_id, timestamp, CAMERA_ID, LOCATION, confidence_score, EVENT_TYPE)
        
        if success:
            track_history[track_id]['attendance_logged'] = True
            attendance_logged_this_session.add(employee_id)  # Mark as logged in session
            
            print(f"✅ ATTENDANCE SUCCESSFULLY LOGGED:")
            print(f"   Employee: {employee_id} ({employee_name})")
            print(f"   Time: {timestamp}")
            print(f"   Track ID: {track_id}")
            print(f"   Frame: {frame_count}")
            print(f"   Session Total: {len(attendance_logged_this_session)} employees logged")
            
            return True, "Successfully logged"
        else:
            print(f"❌ Failed to log attendance: {message}")
            return False, f"Database error: {message}"
    
    return False, "Conditions not met"

def process_video(video_source, frame_interval, similarity_threshold, index, employee_ids):
    global in_count, track_history, employee_persistence, attendance_logged_this_session
    global capture_thread  # FIXED: Add global declaration

    # PROFESSIONAL: Slow video properties for saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    # Verify VideoWriter initialization
    if not out.isOpened():
        print("Error: VideoWriter failed to initialize.")
        exit()
    print(f"✅ VideoWriter initialized with PROFESSIONAL FPS: {FPS}, Size=({WIDTH}x{HEIGHT}), Output={OUTPUT_PATH}")

    # Frame timing for slow processing
    FRAME_DELAY = 1.0 / FPS  # Control processing speed
    last_process_time = time.time()

    # Open CSV file to log the tracking data
    with open('entry_track_log_professional.csv', mode='w', newline='') as csvfile:
        fieldnames = ['Frame', 'TrackID', 'EmployeeID', 'Name', 'Department', 'CenterX', 'CenterY', 
                     'PrevY', 'CurrY', 'LineY', 'Crossed', 'FaceDetected', 'FaceMatched', 
                     'DistanceToLine', 'History', 'AttendanceLogged', 'DetectionMethod', 'LoggingReason', 'UniqueAttendanceCount']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0
        try:
            while True:
                # Control frame processing speed for slow output
                current_time = time.time()
                elapsed = current_time - last_process_time
                
                if elapsed < FRAME_DELAY:
                    time.sleep(FRAME_DELAY - elapsed)
                
                # Get frame from queue
                try:
                    frame = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    print("Warning: Frame queue empty, no new frames received.")
                    continue

                frame_count += 1
                last_process_time = time.time()

                # Validate frame
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                    print(f"Warning: Invalid frame at frame_count {frame_count}")
                    continue

                # Resize frame
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                
                # Create a separate display frame for better visualization
                display_frame = frame.copy()
                
                # Add professional debug visualization first
                add_debug_visualization_professional(display_frame, track_history, ACTUAL_LINE_Y, frame_count)

                # ENHANCED PERSON DETECTION
                boxes, ids, confidences = enhanced_person_detection(frame, model, face_model, frame_count)

                print(f"Frame {frame_count}: Total detected people: {len(boxes)} with IDs: {ids}")

                # Process detections
                if boxes:
                    for i, (box, track_id, confidence) in enumerate(zip(boxes, ids, confidences)):
                        x1, y1, x2, y2 = box
                        x1 = max(0, x1 - 5)
                        y1 = max(0, y1 - 5)
                        x2 = min(WIDTH, x2 + 5)
                        y2 = min(HEIGHT, y2 + 5)
                        if x1 >= x2 or y1 >= y2:
                            continue
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Initialize track_history for new track_id
                        if track_id not in track_history:
                            track_history[track_id] = {
                                'history': [],
                                'crossed': False,
                                'employee_id': None,
                                'face_detected': False,
                                'face_matched': False,
                                'name': None,
                                'department': None,
                                'last_seen_frame': frame_count,
                                'attendance_logged': False,
                                'face_coords': None,
                                'retry_frames': 0,
                                'detection_confidence': confidence,
                                'counted_in_total': False,
                                'detection_method': 'primary' if track_id < 50000 else 'enhanced'
                            }

                        # Update history and track info
                        track_history[track_id]['history'].append(center_y)
                        if len(track_history[track_id]['history']) > 5:
                            track_history[track_id]['history'] = track_history[track_id]['history'][-5:]
                        track_history[track_id]['last_seen_frame'] = frame_count

                        # Line crossing detection
                        crossed, prev_y, curr_y = detect_line_crossing_multiple(
                            track_history, track_id, ACTUAL_LINE_Y, frame_count
                        )
                        
                        if crossed and not track_history[track_id]['crossed']:
                            track_history[track_id]['crossed'] = True
                            track_history[track_id]['retry_frames'] = 30
                            in_count += 1
                            print(f"✅ ENTRY COUNTED: TrackID {track_id} - Total entries = {in_count}")

                        # Face detection - ALWAYS process for visibility
                        should_process_face = True  # Process every frame for visibility
                        
                        if should_process_face:
                            person_crop = frame[y1:y2, x1:x2]
                            if person_crop.size > 0:
                                person_crop_height, person_crop_width = person_crop.shape[:2]

                                faces = detect_faces(person_crop)
                                face_detected = bool(faces)
                                track_history[track_id]['face_detected'] = face_detected

                                if face_detected:
                                    face_crop, (fx1, fy1, fx2, fy2) = faces[0]
                                    fx1 = max(0, fx1)
                                    fy1 = max(0, fy1)
                                    fx2 = min(person_crop_width, fx2)
                                    fy2 = min(person_crop_height, fy2)
                                    
                                    if fx1 < fx2 and fy1 < fy2:
                                        track_history[track_id]['face_coords'] = (fx1, fy1, fx2, fy2)
                                        
                                        # Calculate face coordinates on full frame
                                        face_x1 = x1 + fx1
                                        face_y1 = y1 + fy1
                                        face_x2 = x1 + fx2
                                        face_y2 = y1 + fy2
                                        
                                        # Face matching with FAISS
                                        if index is not None:
                                            embedding = embedder.get_embedding(face_crop)
                                            if embedding is not None:
                                                embedding = embedding / np.linalg.norm(embedding)
                                                D, I = index.search(np.array([embedding]).astype('float32'), 1)
                                                distance = D[0][0]
                                                
                                                if distance < similarity_threshold:
                                                    employee_id = int(employee_ids[I[0][0]])
                                                    name, department = get_employee_details(employee_id)
                                                    
                                                    # Update track history
                                                    track_history[track_id]['employee_id'] = employee_id
                                                    track_history[track_id]['face_matched'] = True
                                                    track_history[track_id]['name'] = name
                                                    track_history[track_id]['department'] = department
                                                    
                                                    # PROFESSIONAL: Draw matched face
                                                    draw_professional_face_detection(display_frame, face_x1, face_y1, 
                                                                                   face_x2, face_y2, "matched", name)
                                                    
                                                    print(f"✅ FACE MATCHED: {name} (ID: {employee_id})")
                                                else:
                                                    # PROFESSIONAL: Draw unknown face
                                                    draw_professional_face_detection(display_frame, face_x1, face_y1, 
                                                                                   face_x2, face_y2, "unknown")
                                        else:
                                            # PROFESSIONAL: Draw detected face
                                            draw_professional_face_detection(display_frame, face_x1, face_y1, 
                                                                           face_x2, face_y2, "detected")

                        # PROFESSIONAL: Draw person detection
                        draw_professional_person_detection(display_frame, track_id, track_history, x1, y1, x2, y2)

                        # Attendance logging
                        attendance_logged, logging_reason = log_employee_attendance(track_id, track_history, frame_count)
                        track_history[track_id]['attendance_logged'] = attendance_logged

                        # CSV logging
                        distance_to_line = abs(center_y - ACTUAL_LINE_Y) if center_y else None
                        history_str = str(track_history[track_id]['history'][-3:])

                        writer.writerow({
                            'Frame': frame_count,
                            'TrackID': track_id,
                            'EmployeeID': track_history[track_id]['employee_id'] if track_history[track_id]['employee_id'] else 'Unknown',
                            'Name': track_history[track_id]['name'] if track_history[track_id]['name'] else 'Unknown',
                            'Department': track_history[track_id]['department'] if track_history[track_id]['department'] else 'Unknown',
                            'CenterX': center_x,
                            'CenterY': center_y,
                            'PrevY': prev_y,
                            'CurrY': curr_y,
                            'LineY': ACTUAL_LINE_Y,
                            'Crossed': track_history[track_id]['crossed'],
                            'FaceDetected': track_history[track_id]['face_detected'],
                            'FaceMatched': track_history[track_id]['face_matched'],
                            'DistanceToLine': distance_to_line,
                            'History': history_str,
                            'AttendanceLogged': attendance_logged,
                            'DetectionMethod': track_history[track_id]['detection_method'],
                            'LoggingReason': logging_reason,
                            'UniqueAttendanceCount': len(attendance_logged_this_session)
                        })

                # Track management
                if frame_count % 30 == 0:
                    manage_crowded_tracks(track_history, frame_count)

                # Save EVERY frame to output video
                if display_frame is not None and display_frame.shape[0] > 0 and display_frame.shape[1] > 0:
                    out.write(display_frame)
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"✅ Frame {frame_count} saved to PROFESSIONAL video (FPS: {FPS})")

                # Display the frame
                cv2.imshow("PROFESSIONAL AI Attendance System", display_frame)

                # ESC to break
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("Program interrupted, cleaning up...")
        finally:
            global stop_thread
            stop_thread = True
            
            # FIXED: Safe thread cleanup
            if 'capture_thread' in globals() and capture_thread.is_alive():
                capture_thread.join(timeout=2.0)
            
            out.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("📊 PROFESSIONAL SESSION SUMMARY")
            print("="*60)
            print(f"✅ Video saved with PROFESSIONAL FPS ({FPS}) to: {OUTPUT_PATH}")
            print(f"✅ Total Entry Count: {in_count}")
            print(f"✅ Unique Employees with Attendance Logged: {len(attendance_logged_this_session)}")
            print(f"✅ CSV log saved to: entry_track_log_professional.csv")
            print("="*60)

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, args=(RTSP_URL,), daemon=True)
capture_thread.start()

def add_debug_visualization_with_crossing_info(frame, track_history, actual_line_y, frame_count):
    """Enhanced debug with crossing information"""
    
    # Call the existing professional visualization
    add_debug_visualization_professional(frame, track_history, actual_line_y, frame_count)
    
    # ADD: Crossing detection status for each active track
    y_pos = 150  # Start below the main stats box
    
    for track_id, data in track_history.items():
        if len(data['history']) > 0 and data.get('face_matched', False):
            # Show detailed info for recognized people
            name = data.get('name', 'Unknown')
            curr_y = data['history'][-1] if data['history'] else 0
            distance_to_line = abs(curr_y - actual_line_y)
            crossed_status = "CROSSED" if data['crossed'] else "NOT CROSSED"
            
            # Color based on status
            color = (0, 255, 0) if data['crossed'] else (0, 255, 255)
            
            status_text = f"{name}: Y={curr_y}, Dist={distance_to_line}, {crossed_status}"
            cv2.putText(frame, status_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_pos += 20
            
            if y_pos > frame.shape[0] - 50:  # Don't go below frame
                break
    
    # ADD: Show line crossing instructions
    instructions = [
        "Press 'W' to move line UP",
        "Press 'S' to move line DOWN", 
        "Press 'R' to RESET line position"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (frame.shape[1] - 250, 30 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def adjust_line_position_realtime(frame, current_line_y):
    """Enhanced real-time line position adjustment"""
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('w') or key == ord('W'):  # Move line up
        current_line_y = max(50, current_line_y - 5)  # Smaller steps
        print(f"📏 Line moved UP to Y={current_line_y}")
    
    elif key == ord('s') or key == ord('S'):  # Move line down
        current_line_y = min(frame.shape[0] - 50, current_line_y + 5)  # Smaller steps
        print(f"📏 Line moved DOWN to Y={current_line_y}")
    
    elif key == ord('q') or key == ord('Q'):  # Big move up
        current_line_y = max(50, current_line_y - 20)
        print(f"📏 Line moved UP (BIG) to Y={current_line_y}")
    
    elif key == ord('a') or key == ord('A'):  # Big move down
        current_line_y = min(frame.shape[0] - 50, current_line_y + 20)
        print(f"📏 Line moved DOWN (BIG) to Y={current_line_y}")
    
    elif key == ord('r') or key == ord('R'):  # Reset to original
        current_line_y = int(frame.shape[0] * 0.4)  # Reset to 40% from top
        print(f"📏 Line RESET to Y={current_line_y}")
    
    return current_line_y

# ENHANCED: Better crossing visualization
def add_enhanced_crossing_debug(frame, track_history, actual_line_y, frame_count):
    """Enhanced visualization showing crossing status and line position info"""
    
    # Call existing professional visualization
    add_debug_visualization_professional(frame, track_history, actual_line_y, frame_count)
    
    # Enhanced line visualization with zones
    line_zone_size = 15
    
    # Draw detection zones
    cv2.line(frame, (0, actual_line_y - line_zone_size), (frame.shape[1], actual_line_y - line_zone_size), (100, 100, 100), 1)  # Upper zone
    cv2.line(frame, (0, actual_line_y + line_zone_size), (frame.shape[1], actual_line_y + line_zone_size), (100, 100, 100), 1)  # Lower zone
    
    # Zone labels
    cv2.putText(frame, "ABOVE ZONE", (10, actual_line_y - line_zone_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(frame, "BELOW ZONE", (10, actual_line_y + line_zone_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Show crossing status for each person
    y_pos = 150
    crossing_count = 0
    
    for track_id, data in track_history.items():
        if len(data['history']) > 0:
            name = data.get('name', f'Track{track_id}')
            curr_y = data['history'][-1]
            distance_to_line = abs(curr_y - actual_line_y)
            crossed_status = "✅ CROSSED" if data['crossed'] else "❌ NOT CROSSED"
            
            if data['crossed']:
                crossing_count += 1
            
            # Color coding
            if data['crossed']:
                color = (0, 255, 0)  # Green
            elif curr_y > actual_line_y:
                color = (0, 255, 255)  # Yellow (below line but not crossed)
            else:
                color = (255, 255, 255)  # White (above line)
            
            status_text = f"{name}: Y={curr_y} D={distance_to_line} {crossed_status}"
            cv2.putText(frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 18
            
            if y_pos > frame.shape[0] - 100:
                break
    
    # Show total crossings prominently
    cv2.rectangle(frame, (frame.shape[1] - 200, 10), (frame.shape[1] - 10, 60), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"CROSSINGS: {crossing_count}", (frame.shape[1] - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL: {in_count}", (frame.shape[1] - 190, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Enhanced control instructions
    instructions = [
        "W/S: Move line UP/DOWN (5px)",
        "Q/A: Move line UP/DOWN (20px)", 
        "R: RESET line position",
        "ESC: Exit"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (frame.shape[1] - 280, frame.shape[0] - 80 + i*15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


# Main
if __name__ == "__main__":

    ACTUAL_LINE_Y = verify_line_position(HEIGHT, ACTUAL_LINE_Y, LINE_Y_RATIO)

    print("🚀 Starting PROFESSIONAL Multi-Person Attendance System...")
    print(f"📏 Entry line position: {ACTUAL_LINE_Y} pixels ({LINE_Y_RATIO*100}% from top)")
    print(f"🎯 Crossing tolerance: ±{CROSSING_TOLERANCE} pixels")
    print(f"📏 VERIFIED Entry line position: {ACTUAL_LINE_Y} pixels ({LINE_Y_RATIO*100}% from top)")
   
    
    employee_embeddings = generate_embeddings()
    index, employee_ids = build_faiss_index()
    
    if index is None or employee_ids is None:
        print("❌ Failed to build FAISS index. Check database and embeddings.")
    else:
        print(f"✅ FAISS index loaded with {len(employee_ids)} employees")
        process_video(
            video_source=RTSP_URL,
            frame_interval=FRAME_INTERVAL,
            similarity_threshold=SIMILARITY_THRESHOLD,
            index=index,
            employee_ids=employee_ids
        )