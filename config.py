"""
Configuration file for AI Attendance System
"""
import os

# Model paths
WEIGHTS_DIR = "./weights"
FACE_DETECTION_MODEL = os.path.join(WEIGHTS_DIR, "det_10g.onnx")
FACE_RECOGNITION_MODEL = os.path.join(WEIGHTS_DIR, "w600k_r50.onnx")

# Face database settings
FACES_DIR = "./faces"
DATABASE_DIR = "./database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "face_database")

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4

# Input settings
INPUT_SIZE = (640, 640)
MAX_FACES_PER_FRAME = 10

# Attendance settings
ATTENDANCE_FILE = "attendance.xlsx"
DUPLICATE_THRESHOLD_HOURS = 8  # Hours before allowing same person to be logged again

# Camera settings
CAMERA_INDEX = 0  # Default webcam

# Display settings
WINDOW_NAME = "AI Attendance System"
DISPLAY_CONFIDENCE = True
DISPLAY_SIMILARITY = True

# Logging
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = "attendance_system.log"