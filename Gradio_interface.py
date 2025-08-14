"""
Gradio Web Interface for AI Attendance System

This provides a web-based interface for the AI attendance system with two main tabs:
1. Add Person to Database - Upload images and add new people
2. Live Attendance View - Real-time webcam monitoring and attendance table

Usage:
    python Gradio_interface.py

Features:
- Real-time face detection and recognition through webcam
- Dynamic attendance table that updates automatically
- Person registration through image upload
- Download attendance Excel file
- Maintains all existing CLI functionality
"""

import gradio as gr
import cv2
import pandas as pd
import numpy as np
import os
import threading
import time
from datetime import datetime
from typing import Optional, Tuple
import logging
import tempfile
import shutil

# Import our existing modules
from main import AttendanceSystemApp
from face_recognition import FaceRecognitionSystem
from attendance_logger import AttendanceLogger
import config


class GradioAttendanceInterface:
    """Gradio web interface for the AI Attendance System."""
    
    def __init__(self):
        """Initialize the Gradio interface."""
        self.app = AttendanceSystemApp()
        self.running = False
        self.current_frame = None
        self.webcam_thread = None
        
        # Initialize logging for the interface
        self.logger = logging.getLogger(__name__)
        
        # Initialize the attendance system
        if not self.app.initialize():
            raise Exception("Failed to initialize attendance system")
            
        self.logger.info("Gradio interface initialized successfully")
    
    def add_person_to_database(self, image_file, person_name: str) -> Tuple[str, str]:
        """
        Add a new person to the face database.
        
        Args:
            image_file: Uploaded image file
            person_name: Name of the person
            
        Returns:
            Tuple of (status_message, updated_database_info)
        """
        try:
            if image_file is None:
                return "❌ Please upload an image first.", self.get_database_info()
            
            if not person_name or not person_name.strip():
                return "❌ Please enter a person's name.", self.get_database_info()
            
            person_name = person_name.strip()
            
            # Create a temporary file for the uploaded image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                # Copy uploaded file to temporary location
                shutil.copy2(image_file, temp_path)
            
            try:
                # Add person to database
                success = self.app.add_person(temp_path, person_name)
                
                if success:
                    # Also save the image to the faces directory for future use
                    faces_image_path = os.path.join(config.FACES_DIR, f"{person_name}.jpg")
                    shutil.copy2(temp_path, faces_image_path)
                    
                    # Restart the face recognition system to include new person
                    self.app.face_system._build_database_from_images()
                    self.app.face_system.save_database()
                    
                    status_msg = f"✅ Successfully added {person_name} to the database!"
                    self.logger.info(f"Added person {person_name} via Gradio interface")
                else:
                    status_msg = f"❌ Failed to add {person_name}. Please ensure the image contains a clear face."
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            return status_msg, self.get_database_info(), self.get_registered_persons()
            
        except Exception as e:
            self.logger.error(f"Error adding person {person_name}: {e}")
            return f"❌ Error adding person: {str(e)}", self.get_database_info(), self.get_registered_persons()
    
    def get_database_info(self) -> str:
        """Get current database information as formatted string."""
        try:
            db_info = self.app.face_system.get_database_info()
            return f"📊 Database: {db_info['total_faces']} faces, {db_info['unique_people']} people"
        except Exception as e:
            return f"❌ Error getting database info: {e}"
    
    def get_registered_persons(self) -> str:
        """Get list of all registered persons in the database."""
        try:
            if not self.app.face_system.face_db or not self.app.face_system.face_db.metadata:
                return "👥 No persons registered yet"
            
            # Get unique names from metadata
            unique_names = list(set(self.app.face_system.face_db.metadata))
            unique_names.sort()  # Sort alphabetically
            
            if not unique_names:
                return "👥 No persons registered yet"
            
            # Format the list
            persons_list = "👥 Registered Persons:\n"
            for i, name in enumerate(unique_names, 1):
                persons_list += f"{i}. {name}\n"
            
            return persons_list.strip()
            
        except Exception as e:
            return f"❌ Error getting persons list: {e}"
    
    def start_webcam_monitoring(self):
        """Start the webcam monitoring in a separate thread."""
        if not self.running:
            # Optimize camera settings for better FPS
            if self.app.cap:
                self.app.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
                self.app.cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
            
            self.running = True
            self.webcam_thread = threading.Thread(target=self._webcam_loop, daemon=True)
            self.webcam_thread.start()
            return "🎥 Webcam monitoring started"
        return "⚠️ Webcam already running"
    
    def stop_webcam_monitoring(self):
        """Stop the webcam monitoring."""
        if self.running:
            self.running = False
            if self.webcam_thread and self.webcam_thread.is_alive():
                self.webcam_thread.join(timeout=2)
            return "⏹️ Webcam monitoring stopped"
        return "⚠️ Webcam not running"
    
    def _webcam_loop(self):
        """Main webcam processing loop running in separate thread."""
        try:
            frame_count = 0
            while self.running and self.app.cap and self.app.cap.isOpened():
                ret, frame = self.app.cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process every frame for display, but do recognition less frequently for performance
                # Detect and recognize faces
                results = self.app.face_system.detect_and_recognize_faces(frame)
                
                # Draw results on frame
                frame = self.app.face_system.draw_results(frame, results)
                
                # Get recognized faces (excluding Unknown)
                recognized_faces = self.app.face_system.get_recognized_faces(results)
                
                # Log attendance for recognized faces
                if recognized_faces:
                    logged_names = self.app.attendance_logger.log_attendance(recognized_faces)
                    
                    # Log info for newly logged attendees
                    for name in logged_names:
                        self.logger.info(f"✓ ATTENDANCE LOGGED: {name}")
                
                # Add system info overlay
                self.app._add_info_overlay(frame, frame_count, len(results), len(recognized_faces))
                
                # Convert BGR to RGB for proper display in web browsers
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Store current frame for display
                self.current_frame = frame_rgb.copy()  # Use copy to prevent threading issues
                
                # Minimal delay for high FPS
                time.sleep(0.005)  # Very small delay
                
        except Exception as e:
            self.logger.error(f"Error in webcam loop: {e}")
        finally:
            self.running = False
    
    def get_webcam_frame(self):
        """Get the current webcam frame for display."""
        if self.current_frame is not None:
            return self.current_frame
        else:
            # Return a placeholder image when webcam is not running
            placeholder = cv2.imread("placeholder.jpg") if os.path.exists("placeholder.jpg") else None
            if placeholder is None:
                # Create a simple placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Click 'Start Webcam' to begin", (120, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Convert BGR to RGB if loading an image file
                placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return placeholder
    
    def get_attendance_data(self) -> pd.DataFrame:
        """Get current attendance data for display."""
        try:
            df = self.app.attendance_logger.get_today_attendance()
            if df.empty:
                # Return empty dataframe with proper columns
                return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Similarity_Score'])
            return df
        except Exception as e:
            self.logger.error(f"Error getting attendance data: {e}")
            return pd.DataFrame(columns=['Name', 'Date', 'Time', 'Similarity_Score'])
    
    def download_attendance_file(self):
        """Return the attendance Excel file for download."""
        if os.path.exists(config.ATTENDANCE_FILE):
            return config.ATTENDANCE_FILE
        else:
            # Create an empty file if it doesn't exist
            empty_df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Similarity_Score'])
            empty_df.to_excel(config.ATTENDANCE_FILE, index=False)
            return config.ATTENDANCE_FILE
    
    def get_today_summary(self) -> str:
        """Get today's attendance summary."""
        try:
            today_df = self.app.attendance_logger.get_today_attendance()
            unique_people = today_df['Name'].nunique() if not today_df.empty else 0
            total_records = len(today_df)
            
            summary = f"📅 Today's Summary:\n"
            summary += f"• Total Records: {total_records}\n"
            summary += f"• Unique People: {unique_people}\n"
            summary += f"• Last Update: {datetime.now().strftime('%H:%M:%S')}"
            
            return summary
        except Exception as e:
            return f"❌ Error getting summary: {e}"
    
    def cleanup(self):
        """Clean up resources when interface is closed."""
        self.stop_webcam_monitoring()
        if self.app:
            self.app.cleanup()


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the interface
    interface = GradioAttendanceInterface()
    
    # Custom CSS for better styling
    css = """
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .webcam-container {
        text-align: center;
    }
    .info-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="AI Attendance System") as demo:
        gr.Markdown("# 🎯 AI Attendance System")
        gr.Markdown("Real-time face recognition and attendance tracking system")
        
        with gr.Tabs():
            # Tab 1: Add Person to Database
            with gr.TabItem("👤 Add Person to Database"):
                gr.Markdown("### Add New Person to Face Recognition Database")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        person_image = gr.File(
                            label="📷 Upload Person's Photo",
                            file_types=["image"],
                            type="filepath"
                        )
                        person_name = gr.Textbox(
                            label="👤 Person's Name",
                            placeholder="Enter the person's full name...",
                            max_lines=1
                        )
                        add_button = gr.Button("➕ Add to Database", variant="primary")
                    
                    with gr.Column(scale=1):
                        add_status = gr.Textbox(
                            label="📊 Status",
                            value="Ready to add new person...",
                            interactive=False
                        )
                        db_info = gr.Textbox(
                            label="🗃️ Database Info",
                            value=interface.get_database_info(),
                            interactive=False
                        )
                        
                        registered_persons = gr.Textbox(
                            label="👥 Registered Persons",
                            value=interface.get_registered_persons(),
                            interactive=False,
                            lines=10
                        )
                
                # Add person functionality
                add_button.click(
                    fn=interface.add_person_to_database,
                    inputs=[person_image, person_name],
                    outputs=[add_status, db_info, registered_persons]
                )
            
            # Tab 2: Live Attendance View
            with gr.TabItem("📹 Live Attendance View"):
                gr.Markdown("### Real-time Face Recognition and Attendance Monitoring")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Webcam controls
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Webcam", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop Webcam", variant="secondary")
                        
                        webcam_status = gr.Textbox(
                            label="📹 Webcam Status",
                            value="Click 'Start Webcam' to begin monitoring",
                            interactive=False
                        )
                        
                        # Webcam display
                        webcam_feed = gr.Image(
                            label="🎥 Live Feed",
                            value=interface.get_webcam_frame(),
                            streaming=True
                        )
                    
                    with gr.Column(scale=1):
                        # Today's summary
                        today_summary = gr.Textbox(
                            label="📊 Today's Summary",
                            value=interface.get_today_summary(),
                            interactive=False,
                            lines=5
                        )
                        
                        # Download button
                        download_btn = gr.DownloadButton(
                            label="📥 Download Attendance Excel",
                            value=interface.download_attendance_file,
                            variant="secondary"
                        )
                
                # Attendance table
                gr.Markdown("### 📋 Today's Attendance Records")
                attendance_table = gr.Dataframe(
                    label="Attendance Log",
                    value=interface.get_attendance_data(),
                    headers=["Name", "Date", "Time", "Similarity Score"],
                    interactive=False,
                    wrap=True
                )
                
                # Webcam control functionality
                start_btn.click(
                    fn=interface.start_webcam_monitoring,
                    outputs=webcam_status
                )
                
                stop_btn.click(
                    fn=interface.stop_webcam_monitoring,
                    outputs=webcam_status
                )
                
                # Auto-refresh components - faster for webcam, slower for data
                webcam_timer = gr.Timer(0.033)  # ~30 FPS for webcam display
                webcam_timer.tick(
                    fn=interface.get_webcam_frame,
                    outputs=webcam_feed
                )
                
                data_timer = gr.Timer(2)  # 2 seconds for data updates
                data_timer.tick(
                    fn=lambda: [
                        interface.get_attendance_data(),
                        interface.get_today_summary()
                    ],
                    outputs=[attendance_table, today_summary]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("💡 **Tips:**")
        gr.Markdown("• Ensure good lighting for better face detection")
        gr.Markdown("• Upload clear, front-facing photos when adding new people")
        gr.Markdown("• The system prevents duplicate attendance within 8 hours")
        gr.Markdown("• All data is automatically saved to attendance.xlsx")
    
    return demo, interface


def main():
    """Main entry point for the Gradio interface."""
    try:
        print("🚀 Starting AI Attendance System - Gradio Interface...")
        print("📋 Initializing components...")
        
        # Create the interface
        demo, interface_obj = create_gradio_interface()
        
        print("✅ Interface ready!")
        print("🌐 Starting web server...")
        print("📖 Open your browser and go to the URL shown below")
        print("⚠️  Press Ctrl+C to stop the server")
        
        # Launch the interface
        demo.launch(
            server_name="0.0.0.0",  # Allow access from any IP
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True if you want public sharing
            show_api=False,         # Hide API docs
            show_error=True,        # Show detailed errors
            inbrowser=True          # Auto-open browser
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        logging.error(f"Error starting Gradio interface: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            if 'interface_obj' in locals():
                interface_obj.cleanup()
        except:
            pass
        print("🔄 Cleanup completed")
        print("👋 Thank you for using AI Attendance System!")


if __name__ == "__main__":
    main()