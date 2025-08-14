"""
Face Recognition Module for AI Attendance System

This module handles face detection, recognition, and identification using:
- SCRFD for face detection
- ArcFace for face recognition
- FAISS for face identification
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from datetime import datetime

from models import SCRFD, ArcFace
from database import FaceDatabase
from utils.helpers import draw_bbox_info, draw_bbox
import config


class FaceRecognitionSystem:
    """Main class for handling all face recognition operations."""
    
    def __init__(self):
        """Initialize the face recognition system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.detector = None
        self.recognizer = None
        self.face_db = None
        
        # Color mapping for different people
        self.colors = {}
        
        self._initialize_models()
        self._initialize_database()
        
    def _initialize_models(self):
        """Initialize face detection and recognition models."""
        try:
            self.logger.info("Initializing face detection model...")
            self.detector = SCRFD(
                model_path=config.FACE_DETECTION_MODEL,
                input_size=config.INPUT_SIZE,
                conf_thres=config.CONFIDENCE_THRESHOLD,
                iou_thres=config.IOU_THRESHOLD
            )
            
            self.logger.info("Initializing face recognition model...")
            self.recognizer = ArcFace(config.FACE_RECOGNITION_MODEL)
            
            self.logger.info("Models initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize or load the face database."""
        try:
            self.logger.info("Initializing face database...")
            self.face_db = FaceDatabase(db_path=config.DATABASE_PATH)
            
            # Try to load existing database
            if self.face_db.load():
                self.logger.info(f"Loaded existing database with {self.face_db.index.ntotal} faces")
            else:
                self.logger.info("No existing database found, will build from faces directory")
                self._build_database_from_images()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _build_database_from_images(self):
        """Build the face database from images in the faces directory."""
        if not os.path.exists(config.FACES_DIR):
            self.logger.warning(f"Faces directory {config.FACES_DIR} does not exist")
            return
            
        self.logger.info("Building face database from images...")
        face_count = 0
        
        for filename in os.listdir(config.FACES_DIR):
            if not (filename.lower().endswith('.jpg') or 
                   filename.lower().endswith('.png') or 
                   filename.lower().endswith('.jpeg')):
                continue
                
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(config.FACES_DIR, filename)
            
            if self.add_person_to_database(image_path, name):
                face_count += 1
                
        self.logger.info(f"Added {face_count} faces to database")
        if face_count > 0:
            self.face_db.save()
    
    def add_person_to_database(self, image_path: str, name: str) -> bool:
        """
        Add a person to the face database.
        
        Args:
            image_path: Path to the person's image
            name: Name of the person
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return False
            
            # Detect faces in the image
            bboxes, kpss = self.detector.detect(image, max_num=1)
            
            if len(kpss) == 0:
                self.logger.warning(f"No face detected in {image_path}")
                return False
            
            # Get face embedding
            embedding = self.recognizer.get_embedding(image, kpss[0])
            
            # Add to database
            self.face_db.add_face(embedding, name)
            self.logger.info(f"Added {name} to database")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding person {name}: {e}")
            return False
    
    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: Input frame/image
            
        Returns:
            List of tuples containing (name, similarity, bbox) for each detected face
        """
        try:
            # Detect faces
            bboxes, kpss = self.detector.detect(frame, max_num=config.MAX_FACES_PER_FRAME)
            
            results = []
            
            for bbox, kps in zip(bboxes, kpss):
                # Extract confidence score
                *bbox_coords, conf_score = bbox.astype(np.int32)
                bbox_coords = np.array(bbox_coords)
                
                # Get face embedding
                embedding = self.recognizer.get_embedding(frame, kps)
                
                # Search in database
                name, similarity = self.face_db.search(embedding, config.SIMILARITY_THRESHOLD)
                
                results.append((name, similarity, bbox_coords))
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in face detection/recognition: {e}")
            return []
    
    def draw_results(self, frame: np.ndarray, results: List[Tuple[str, float, np.ndarray]]) -> np.ndarray:
        """
        Draw bounding boxes and recognition results on the frame.
        
        Args:
            frame: Input frame
            results: List of recognition results
            
        Returns:
            Frame with drawn results
        """
        for name, similarity, bbox in results:
            if name != "Unknown":
                # Assign a consistent color for each person
                if name not in self.colors:
                    self.colors[name] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                
                # Draw bounding box with person info
                draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=self.colors[name])
            else:
                # Draw red bounding box for unknown faces
                draw_bbox(frame, bbox, (0, 0, 255))
        
        return frame
    
    def get_recognized_faces(self, results: List[Tuple[str, float, np.ndarray]]) -> List[Tuple[str, float]]:
        """
        Get list of recognized faces (excluding 'Unknown').
        
        Args:
            results: List of recognition results
            
        Returns:
            List of tuples containing (name, similarity) for recognized faces only
        """
        recognized = []
        for name, similarity, _ in results:
            if name != "Unknown":
                recognized.append((name, similarity))
        return recognized
    
    def save_database(self):
        """Save the current face database to disk."""
        if self.face_db:
            self.face_db.save()
            self.logger.info("Face database saved successfully")
    
    def get_database_info(self) -> dict:
        """
        Get information about the current face database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.face_db:
            return {"total_faces": 0, "unique_people": 0}
        
        return {
            "total_faces": self.face_db.index.ntotal,
            "unique_people": len(set(self.face_db.metadata)) if self.face_db.metadata else 0
        }