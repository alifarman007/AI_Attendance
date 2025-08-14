"""
Attendance Logger Module for AI Attendance System

This module handles logging attendance to Excel files and prevents duplicate entries.
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Set
import config


class AttendanceLogger:
    """Handles attendance logging and Excel file operations."""
    
    def __init__(self):
        """Initialize the attendance logger."""
        self.logger = logging.getLogger(__name__)
        self.attendance_file = config.ATTENDANCE_FILE
        
        # Keep track of recent attendees to prevent duplicates
        self.recent_attendees = {}  # {name: last_logged_time}
        
        self._load_existing_attendance()
    
    def _load_existing_attendance(self):
        """Load existing attendance data to check for recent entries."""
        try:
            if os.path.exists(self.attendance_file):
                df = pd.read_excel(self.attendance_file)
                
                # Load recent attendees from today's records
                today = datetime.now().date()
                today_records = df[pd.to_datetime(df['Date']).dt.date == today]
                
                for _, row in today_records.iterrows():
                    name = row['Name']
                    time_str = row['Time']
                    
                    # Parse time and create full datetime
                    time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                    full_datetime = datetime.combine(today, time_obj)
                    
                    # Store in recent attendees
                    if name not in self.recent_attendees or full_datetime > self.recent_attendees[name]:
                        self.recent_attendees[name] = full_datetime
                
                self.logger.info(f"Loaded {len(today_records)} attendance records from today")
            else:
                self.logger.info("No existing attendance file found, will create new one")
                
        except Exception as e:
            self.logger.error(f"Error loading existing attendance: {e}")
    
    def _should_log_attendance(self, name: str) -> bool:
        """
        Check if attendance should be logged for this person.
        
        Args:
            name: Person's name
            
        Returns:
            bool: True if should log, False if too recent
        """
        if name not in self.recent_attendees:
            return True
        
        last_logged = self.recent_attendees[name]
        current_time = datetime.now()
        
        # Check if enough time has passed since last log
        time_diff = current_time - last_logged
        threshold = timedelta(hours=config.DUPLICATE_THRESHOLD_HOURS)
        
        return time_diff >= threshold
    
    def log_attendance(self, recognized_faces: List[Tuple[str, float]]) -> List[str]:
        """
        Log attendance for recognized faces.
        
        Args:
            recognized_faces: List of tuples containing (name, similarity_score)
            
        Returns:
            List of names that were actually logged (excluding duplicates)
        """
        logged_names = []
        current_time = datetime.now()
        current_date = current_time.date()
        current_time_str = current_time.strftime('%H:%M:%S')
        
        new_entries = []
        
        for name, similarity in recognized_faces:
            if self._should_log_attendance(name):
                # Log this person's attendance
                new_entries.append({
                    'Name': name,
                    'Date': current_date,
                    'Time': current_time_str,
                    'Similarity_Score': round(similarity, 3)
                })
                
                # Update recent attendees
                self.recent_attendees[name] = current_time
                logged_names.append(name)
                
                self.logger.info(f"Logged attendance for {name} at {current_time_str}")
            else:
                last_time = self.recent_attendees[name]
                time_since = current_time - last_time
                hours_since = time_since.total_seconds() / 3600
                
                self.logger.debug(f"Skipped {name} - last logged {hours_since:.1f} hours ago")
        
        # Save new entries to Excel if any
        if new_entries:
            self._save_to_excel(new_entries)
        
        return logged_names
    
    def _save_to_excel(self, new_entries: List[dict]):
        """
        Save new attendance entries to Excel file.
        
        Args:
            new_entries: List of new attendance records
        """
        try:
            # Create DataFrame from new entries
            new_df = pd.DataFrame(new_entries)
            
            # Load existing data if file exists
            if os.path.exists(self.attendance_file):
                existing_df = pd.read_excel(self.attendance_file)
                # Combine existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Sort by date and time
            combined_df['DateTime'] = pd.to_datetime(
                combined_df['Date'].astype(str) + ' ' + combined_df['Time']
            )
            combined_df = combined_df.sort_values('DateTime')
            combined_df = combined_df.drop('DateTime', axis=1)
            
            # Save to Excel
            combined_df.to_excel(self.attendance_file, index=False)
            
            self.logger.info(f"Saved {len(new_entries)} new attendance records to {self.attendance_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving to Excel: {e}")
    
    def get_today_attendance(self) -> pd.DataFrame:
        """
        Get today's attendance records.
        
        Returns:
            DataFrame with today's attendance
        """
        try:
            if not os.path.exists(self.attendance_file):
                return pd.DataFrame()
            
            df = pd.read_excel(self.attendance_file)
            today = datetime.now().date()
            
            # Filter for today's records
            today_records = df[pd.to_datetime(df['Date']).dt.date == today]
            
            return today_records.sort_values('Time')
            
        except Exception as e:
            self.logger.error(f"Error getting today's attendance: {e}")
            return pd.DataFrame()
    
    def get_attendance_summary(self, days: int = 7) -> dict:
        """
        Get attendance summary for the last N days.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary with attendance statistics
        """
        try:
            if not os.path.exists(self.attendance_file):
                return {"total_records": 0, "unique_people": 0, "days_covered": 0}
            
            df = pd.read_excel(self.attendance_file)
            
            # Filter for last N days
            cutoff_date = datetime.now().date() - timedelta(days=days-1)
            recent_records = df[pd.to_datetime(df['Date']).dt.date >= cutoff_date]
            
            return {
                "total_records": len(recent_records),
                "unique_people": recent_records['Name'].nunique(),
                "days_covered": recent_records['Date'].nunique(),
                "date_range": f"{cutoff_date} to {datetime.now().date()}"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting attendance summary: {e}")
            return {"error": str(e)}
    
    def export_attendance(self, start_date: str = None, end_date: str = None, 
                         output_file: str = None) -> bool:
        """
        Export attendance data for a specific date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            output_file: Output file name (default: attendance_export_YYYYMMDD.xlsx)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.attendance_file):
                self.logger.error("No attendance file found")
                return False
            
            df = pd.read_excel(self.attendance_file)
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date).date()
                df = df[pd.to_datetime(df['Date']).dt.date >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date).date()
                df = df[pd.to_datetime(df['Date']).dt.date <= end_date]
            
            # Generate output filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d')
                output_file = f"attendance_export_{timestamp}.xlsx"
            
            # Save exported data
            df.to_excel(output_file, index=False)
            
            self.logger.info(f"Exported {len(df)} records to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting attendance: {e}")
            return False