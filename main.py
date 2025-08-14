"""
AI Attendance System - Main Application

This is the main entry point for the AI attendance system that uses face recognition
to automatically log attendance when people appear in front of the camera.

Features:
- Real-time face detection using SCRFD
- Face recognition using ArcFace
- FAISS-based face identification  
- Automatic attendance logging to Excel
- Duplicate prevention within the same day
- Person registration system
"""

import cv2
import sys
import logging
import argparse
import signal
from datetime import datetime
from pathlib import Path

from face_recognition import FaceRecognitionSystem
from attendance_logger import AttendanceLogger
from utils.logging import setup_logging
import config


class AttendanceSystemApp:
    """Main application class for the AI Attendance System."""
    
    def __init__(self):
        """Initialize the attendance system application."""
        self.running = False
        self.face_system = None
        self.attendance_logger = None
        self.cap = None
        
        # Setup logging
        setup_logging(log_to_file=config.LOG_TO_FILE)
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _check_dependencies(self) -> bool:
        """Check if all required model files and directories exist."""
        required_files = [
            config.FACE_DETECTION_MODEL,
            config.FACE_RECOGNITION_MODEL
        ]
        
        required_dirs = [
            config.WEIGHTS_DIR,
            config.DATABASE_DIR,
            config.FACES_DIR
        ]
        
        # Check directories
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
        
        # Check model files
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.logger.error("Missing required model files:")
            for file_path in missing_files:
                self.logger.error(f"  - {file_path}")
            self.logger.error("Please download the required model files to the weights/ directory")
            return False
        
        return True
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing AI Attendance System...")
            
            # Check dependencies
            if not self._check_dependencies():
                return False
            
            # Initialize face recognition system
            self.logger.info("Loading face recognition models...")
            self.face_system = FaceRecognitionSystem()
            
            # Initialize attendance logger
            self.logger.info("Setting up attendance logger...")
            self.attendance_logger = AttendanceLogger()
            
            # Initialize camera
            self.logger.info(f"Connecting to camera {config.CAMERA_INDEX}...")
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {config.CAMERA_INDEX}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    def add_person(self, image_path: str, name: str) -> bool:
        """
        Add a new person to the face database.
        
        Args:
            image_path: Path to the person's photo
            name: Person's name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.face_system:
            self.logger.error("Face recognition system not initialized")
            return False
        
        success = self.face_system.add_person_to_database(image_path, name)
        if success:
            self.face_system.save_database()
            self.logger.info(f"Successfully added {name} to the system")
        
        return success
    
    def run_attendance_monitoring(self):
        """Run the main attendance monitoring loop."""
        if not self.initialize():
            return False
        
        self.running = True
        frame_count = 0
        last_log_time = datetime.now()
        
        self.logger.info("Starting attendance monitoring...")
        self.logger.info("Press 'q' to quit, 'a' to add person, 's' for statistics")
        
        # Display initial system info
        db_info = self.face_system.get_database_info()
        self.logger.info(f"Database loaded: {db_info['total_faces']} faces, {db_info['unique_people']} unique people")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Detect and recognize faces
                results = self.face_system.detect_and_recognize_faces(frame)
                
                # Draw results on frame
                frame = self.face_system.draw_results(frame, results)
                
                # Get recognized faces (excluding Unknown)
                recognized_faces = self.face_system.get_recognized_faces(results)
                
                # Log attendance for recognized faces
                if recognized_faces:
                    logged_names = self.attendance_logger.log_attendance(recognized_faces)
                    
                    # Display notification for newly logged attendees
                    for name in logged_names:
                        self.logger.info(f"✓ ATTENDANCE LOGGED: {name}")
                
                # Add system info overlay
                self._add_info_overlay(frame, frame_count, len(results), len(recognized_faces))
                
                # Display frame
                cv2.imshow(config.WINDOW_NAME, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested by user")
                    break
                elif key == ord('a'):
                    self._handle_add_person()
                elif key == ord('s'):
                    self._show_statistics()
                
                # Log periodic statistics
                current_time = datetime.now()
                if (current_time - last_log_time).seconds >= 60:  # Every minute
                    today_count = len(self.attendance_logger.get_today_attendance())
                    self.logger.info(f"System running - {today_count} attendance records today")
                    last_log_time = current_time
        
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        
        finally:
            self.cleanup()
    
    def _add_info_overlay(self, frame, frame_count, total_faces, recognized_faces):
        """Add system information overlay to the frame."""
        h, w = frame.shape[:2]
        
        # Create info text
        info_lines = [
            f"Frame: {frame_count}",
            f"Faces: {total_faces}",
            f"Recognized: {recognized_faces}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        # Draw info background
        cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 100), (255, 255, 255), 1)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y = 30 + i * 20
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _handle_add_person(self):
        """Handle interactive person addition."""
        print("\n=== Add New Person ===")
        print("1. Save an image of the person in the 'faces/' directory")
        print("2. Name the image file with the person's name (e.g., 'john_doe.jpg')")
        print("3. Press Enter to rebuild the database, or 'c' to cancel")
        
        response = input("Press Enter to continue or 'c' to cancel: ").strip().lower()
        
        if response != 'c':
            try:
                # Rebuild database from faces directory
                self.face_system._build_database_from_images()
                self.face_system.save_database()
                print("Database updated successfully!")
                self.logger.info("Database manually updated")
            except Exception as e:
                print(f"Error updating database: {e}")
                self.logger.error(f"Error updating database: {e}")
    
    def _show_statistics(self):
        """Show attendance statistics."""
        print("\n=== Attendance Statistics ===")
        
        # Today's attendance
        today_df = self.attendance_logger.get_today_attendance()
        print(f"Today's Attendance ({len(today_df)} records):")
        if not today_df.empty:
            for _, row in today_df.iterrows():
                print(f"  {row['Name']} at {row['Time']} (similarity: {row['Similarity_Score']:.3f})")
        else:
            print("  No records yet today")
        
        # Weekly summary
        weekly_summary = self.attendance_logger.get_attendance_summary(days=7)
        print(f"\nWeekly Summary ({weekly_summary.get('date_range', 'N/A')}):")
        print(f"  Total records: {weekly_summary.get('total_records', 0)}")
        print(f"  Unique people: {weekly_summary.get('unique_people', 0)}")
        print(f"  Days covered: {weekly_summary.get('days_covered', 0)}")
        
        # Database info
        db_info = self.face_system.get_database_info()
        print(f"\nDatabase Info:")
        print(f"  Total faces in database: {db_info['total_faces']}")
        print(f"  Unique people: {db_info['unique_people']}")
        
        print("\nPress any key to continue...")
        cv2.waitKey(0)
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.face_system:
            self.face_system.save_database()
        
        self.logger.info("Cleanup completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Attendance System")
    
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=config.CAMERA_INDEX,
        help="Camera index (default: 0)"
    )
    
    parser.add_argument(
        "--add-person",
        nargs=2,
        metavar=("IMAGE_PATH", "NAME"),
        help="Add a person to database: --add-person path/to/image.jpg 'Person Name'"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export attendance data to Excel"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show attendance statistics and exit"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Update camera index if provided
    if args.camera != config.CAMERA_INDEX:
        config.CAMERA_INDEX = args.camera
    
    app = AttendanceSystemApp()
    
    try:
        if args.add_person:
            # Add person mode
            image_path, name = args.add_person
            print(f"Adding person: {name} from {image_path}")
            
            if app.initialize():
                success = app.add_person(image_path, name)
                print("Success!" if success else "Failed!")
                app.cleanup()
            
        elif args.export:
            # Export mode
            if app.initialize():
                success = app.attendance_logger.export_attendance()
                print("Export completed!" if success else "Export failed!")
                app.cleanup()
            
        elif args.stats:
            # Statistics mode
            if app.initialize():
                app._show_statistics()
                app.cleanup()
        
        else:
            # Normal monitoring mode
            print("Starting AI Attendance System...")
            print("Make sure you have:")
            print("1. Model files in weights/ directory:")
            print("   - det_10g.onnx (face detection)")
            print("   - w600k_r50.onnx (face recognition)")
            print("2. Person photos in faces/ directory (named as person_name.jpg)")
            print()
            
            app.run_attendance_monitoring()
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Application error: {e}")
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        print("Thank you for using AI Attendance System!")


if __name__ == "__main__":
    main()