"""
CSV Logger Module
Real-time data logging for face tracking with research-style temporal log format
"""

import csv
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import json
import threading
from collections import deque

class FaceTrackingLogger:
    """Real-time CSV logger for face tracking data"""
    
    def __init__(self, output_file: str = "realtime_face_heatmap_log.csv", buffer_size: int = 50):
        """
        Initialize the logger
        
        Args:
            output_file: Path to output CSV file
            buffer_size: Number of rows to buffer before writing to disk (smaller for performance)
        """
        self.output_file = output_file
        self.buffer_size = buffer_size
        
        # CSV headers with enhanced schema (Phase 7 fix)
        self.headers = [
            'timestamp',
            'dominant_region',
            'region_confidence_score',
            'persistence_duration_seconds',
            'heatmap_centroid_x',
            'heatmap_centroid_y',
            'region_switch_flag',
            'smoothed_score',
            'raw_score',
            'RecordingTime_ms',
            'Time_of_Day_hms_ms',
            'Frame_Number',
            'Camera_Type',
            'Top_Region_5s',
            'Top_Region_10s',
            'Top_Region_15s',
            'Region_Activity_Scores_JSON',
            'Heatmap_Intensity_Max',
            'Heatmap_Mean_Intensity',
            'Heatmap_Active_Area_Ratio',
            'Depth_Avg_Meters',
            'Depth_Min_Meters',
            'Depth_Max_Meters',
            'Depth_Std_Meters',
            'Face_Detected'
        ]
        
        # Data buffer for batch writing
        self.data_buffer = deque(maxlen=buffer_size)
        
        # Row counter and switch tracking (Phase 7 fix)
        self.row_counter = 0
        self.last_dominant_region = None
        self.region_switch_start_time = None
        
        # Start time for recording time calculation
        self.start_time = time.time()
        
        # Thread lock for thread-safe writing
        self.write_lock = threading.Lock()
        
        # Initialize CSV file
        self._initialize_csv_file()
    
    def _initialize_csv_file(self):
        """Initialize CSV file with headers"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        # Write headers
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
    
    def log_frame_data(self, tracking_result: Dict[str, Any]):
        """
        Log tracking data for a single frame with enhanced schema (Phase 7 fix)
        
        Args:
            tracking_result: Dictionary containing tracking results from FaceTracker
        """
        current_time = time.time()
        
        # Calculate recording time in milliseconds
        recording_time_ms = int((current_time - self.start_time) * 1000)
        
        # Format time of day
        time_of_day = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
        
        # Extract dominant regions for time windows
        dominant_regions = tracking_result.get('dominant_regions_windows', {})
        top_region_5s = dominant_regions.get('5s', 'None')
        top_region_10s = dominant_regions.get('10s', 'None')
        top_region_15s = dominant_regions.get('15s', 'None')
        
        # Convert region objects to strings
        current_dominant = tracking_result.get('dominant_region')
        if current_dominant is not None:
            current_dominant_str = current_dominant.value if hasattr(current_dominant, 'value') else str(current_dominant)
        else:
            current_dominant_str = 'None'
        
        # Phase 7 fix: Calculate enhanced metrics
        # Region confidence score (normalized smoothed score)
        activity_scores = tracking_result.get('activity_scores', {})
        smoothed_scores = tracking_result.get('smoothed_activity_scores', {})
        
        if current_dominant in smoothed_scores:
            confidence_score = smoothed_scores[current_dominant]
            raw_score = activity_scores.get(current_dominant, 0.0)
        else:
            confidence_score = 0.0
            raw_score = 0.0
        
        # Persistence duration calculation
        if current_dominant != self.last_dominant_region:
            # Region switch detected
            self.region_switch_start_time = current_time
            persistence_duration = 0.0
            region_switch_flag = 1
        else:
            # Same region continues
            if self.region_switch_start_time is not None:
                persistence_duration = current_time - self.region_switch_start_time
            else:
                persistence_duration = 0.0
            region_switch_flag = 0
        
        self.last_dominant_region = current_dominant
        
        # Convert other regions to strings
        def region_to_str(region):
            if region is not None:
                return region.value if hasattr(region, 'value') else str(region)
            return 'None'
        
        top_region_5s_str = region_to_str(top_region_5s)
        top_region_10s_str = region_to_str(top_region_10s)
        top_region_15s_str = region_to_str(top_region_15s)
        
        # Get region activity scores as JSON
        activity_scores_json = json.dumps({
            region.value if hasattr(region, 'value') else str(region): float(score)
            for region, score in activity_scores.items()
        })
        
        # Get heatmap statistics
        heatmap_stats = tracking_result.get('heatmap_stats', {})
        heatmap_intensity_max = heatmap_stats.get('max_intensity', 0.0)
        heatmap_mean_intensity = heatmap_stats.get('mean_intensity', 0.0)
        heatmap_active_area_ratio = heatmap_stats.get('active_area_ratio', 0.0)
        heatmap_centroid_x = heatmap_stats.get('centroid_x', 0)
        heatmap_centroid_y = heatmap_stats.get('centroid_y', 0)
        
        # Get depth statistics (if available)
        depth_stats = tracking_result.get('depth_stats', {})
        depth_avg = depth_stats.get('avg_meters', 0.0)
        depth_min = depth_stats.get('min_meters', 0.0)
        depth_max = depth_stats.get('max_meters', 0.0)
        depth_std = depth_stats.get('std_meters', 0.0)
        
        # Get camera type
        camera_type = tracking_result.get('camera_type', 'webcam')
        
        # Face detected flag
        face_detected = tracking_result.get('face_detected', False)
        
        # Create enhanced row data (Phase 7 fix)
        row_data = [
            current_time,  # timestamp
            current_dominant_str,  # dominant_region
            confidence_score,  # region_confidence_score
            persistence_duration,  # persistence_duration_seconds
            heatmap_centroid_x,  # heatmap_centroid_x
            heatmap_centroid_y,  # heatmap_centroid_y
            region_switch_flag,  # region_switch_flag
            confidence_score,  # smoothed_score
            raw_score,  # raw_score
            recording_time_ms,
            time_of_day,
            tracking_result.get('frame_number', self.row_counter),
            camera_type,
            top_region_5s_str,
            top_region_10s_str,
            top_region_15s_str,
            activity_scores_json,
            heatmap_intensity_max,
            heatmap_mean_intensity,
            heatmap_active_area_ratio,
            depth_avg,
            depth_min,
            depth_max,
            depth_std,
            str(face_detected)
        ]
        
        # Add to buffer
        with self.write_lock:
            self.data_buffer.append(row_data)
            self.row_counter += 1
        
        # Write to disk if buffer is full
        if len(self.data_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered data to CSV file"""
        if not self.data_buffer:
            return
        
        with self.write_lock:
            try:
                with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(self.data_buffer)
                
                # Clear buffer
                self.data_buffer.clear()
                
            except Exception as e:
                print(f"Error writing to CSV file: {e}")
    
    def force_write(self):
        """Force write all buffered data to disk"""
        self._flush_buffer()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the logged data
        
        Returns:
            Dictionary containing log statistics
        """
        if not os.path.exists(self.output_file):
            return {
                'total_rows': 0,
                'file_size_bytes': 0,
                'recording_duration_ms': 0,
                'frames_with_face': 0,
                'frames_without_face': 0
            }
        
        # Count total rows (excluding header)
        with open(self.output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            total_rows = sum(1 for row in reader) - 1  # Subtract header
        
        # Get file size
        file_size = os.path.getsize(self.output_file)
        
        # Calculate recording duration
        current_time = time.time()
        recording_duration_ms = int((current_time - self.start_time) * 1000)
        
        # Count face detection (this would require reading the file, simplified here)
        frames_with_face = 0  # Would need to parse CSV for accurate count
        frames_without_face = 0
        
        return {
            'total_rows': total_rows,
            'file_size_bytes': file_size,
            'recording_duration_ms': recording_duration_ms,
            'frames_with_face': frames_with_face,
            'frames_without_face': frames_without_face,
            'current_buffer_size': len(self.data_buffer)
        }
    
    def export_summary_report(self, output_file: str = None):
        """
        Export a summary report of the logging session
        
        Args:
            output_file: Optional output file for the report
        """
        if output_file is None:
            output_file = self.output_file.replace('.csv', '_summary.txt')
        
        stats = self.get_log_statistics()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Face Tracking Session Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Output File: {self.output_file}\n")
            f.write(f"Total Frames Logged: {stats['total_rows']}\n")
            f.write(f"Recording Duration: {stats['recording_duration_ms'] / 1000:.2f} seconds\n")
            f.write(f"File Size: {stats['file_size_bytes'] / 1024:.2f} KB\n")
            f.write(f"Average FPS: {stats['total_rows'] / (stats['recording_duration_ms'] / 1000):.2f}\n")
            f.write(f"Buffer Status: {stats['current_buffer_size']}/{self.buffer_size}\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def reset(self):
        """Reset the logger for a new session"""
        # Force write any remaining data
        self.force_write()
        
        # Reset counters
        self.row_counter = 0
        self.start_time = time.time()
        self.data_buffer.clear()
        
        # Reinitialize CSV file
        self._initialize_csv_file()
    
    def set_output_file(self, new_file: str):
        """
        Change the output file
        
        Args:
            new_file: New output file path
        """
        # Write any pending data to old file
        self.force_write()
        
        # Update file path
        self.output_file = new_file
        
        # Initialize new file
        self._initialize_csv_file()
    
    def __del__(self):
        """Destructor to ensure data is written"""
        self.force_write()
