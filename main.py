"""
Main Application Module
Real-time face heatmap tracking system with OpenCV visualization
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys
import logging
from datetime import datetime
from typing import Optional

from tracker import FaceTracker
from logger import FaceTrackingLogger
from regions import FacialRegion
from facemap import FaceMapOverlay
from camera_manager import CameraManager
import config

class FaceHeatmapApp:
    """Main application class for real-time face heatmap tracking"""
    
    def __init__(self, camera_index: int = None, camera_manager: CameraManager = None, 
                 save_video: bool = False, output_dir: str = "output", show_landmarks: bool = True,
                 show_heatmap: bool = True, show_regions: bool = True):
        """
        Initialize the face heatmap application
        
        Args:
            camera_index: Camera device index (legacy support)
            camera_manager: CameraManager instance (preferred)
            save_video: Whether to save video output
            output_dir: Directory for output files
            show_landmarks: Whether to show face mesh landmarks
            show_heatmap: Whether to show heatmap overlay
            show_regions: Whether to show region labels
        """
        # Handle camera initialization - prefer CameraManager, fallback to camera_index
        if camera_manager is not None:
            self.camera_manager = camera_manager
            self.camera_index = None
        else:
            self.camera_manager = CameraManager()
            if camera_index is not None:
                if not self.camera_manager.initialize('webcam', camera_index):
                    print(f"Error: Could not initialize webcam at index {camera_index}")
                else:
                    self.camera_index = camera_index
            else:
                # Default to webcam index 0
                if not self.camera_manager.initialize('webcam', 0):
                    print("Error: Could not initialize default webcam")
                else:
                    self.camera_index = 0
        
        self.save_video = save_video
        self.output_dir = output_dir
        self.show_landmarks = show_landmarks
        self.show_heatmap = show_heatmap
        self.show_regions = show_regions
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.config_manager = config.ConfigManager()
        self.tracker = FaceTracker()
        self.facemap = FaceMapOverlay(self.config_manager)
        self.logger = FaceTrackingLogger(
            output_file=os.path.join(output_dir, "realtime_face_heatmap_log.csv")
        )
        
        # Video capture
        self.cap = None
        self.video_writer = None
        
        # Application state
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display settings
        self.window_name = "Face Heatmap Tracking"
        self.display_width = 640    # Match camera resolution for performance
        self.display_height = 480
        
        # Colors for visualization
        self.colors = {
            'text': (255, 255, 255),           # White
            'background': (0, 0, 0),           # Black
            'landmark': (0, 255, 0),          # Green
            'bounding_box': (255, 0, 0),      # Red
            'region_text': (255, 255, 0)      # Yellow
        }
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture - now uses CameraManager"""
        try:
            # CameraManager is already initialized in __init__
            if self.camera_manager and self.camera_manager.is_initialized:
                print(f"Camera initialized successfully: {self.camera_manager.get_camera_type()}")
                return True
            else:
                print("Error: CameraManager not initialized")
                return False
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def initialize_video_writer(self, frame_shape):
        """Initialize video writer for saving output"""
        if not self.save_video:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = os.path.join(self.output_dir, f"face_heatmap_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        
        self.video_writer = cv2.VideoWriter(
            video_file, fourcc, fps, 
            (frame_shape[1], frame_shape[0])
        )
        
        if self.video_writer.isOpened():
            print(f"Video writer initialized: {video_file}")
        else:
            print("Warning: Could not initialize video writer")
            self.save_video = False
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw face mesh landmarks on frame"""
        if landmarks is None or not self.show_landmarks:
            return frame
        
        # Draw landmarks as small circles
        for i, (x, y) in enumerate(landmarks):
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 2, self.colors['landmark'], -1)
        
        return frame
    
    def draw_face_bounding_box(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw bounding box around detected face"""
        if landmarks is None:
            return frame
        
        x_min, y_min, x_max, y_max = self.tracker.get_face_bounding_box(landmarks)
        
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                     self.colors['bounding_box'], 2)
        
        return frame
    
    def draw_region_labels(self, frame: np.ndarray, tracking_result: dict) -> np.ndarray:
        """Draw region activity information on frame"""
        if not self.show_regions or not tracking_result.get('face_detected'):
            return frame
        
        # Get current dominant region
        dominant_region = tracking_result.get('dominant_region')
        dominant_regions_windows = tracking_result.get('dominant_regions_windows', {})
        
        # Prepare text
        y_offset = 30
        line_height = 25
        
        # Current dominant region
        if dominant_region:
            text = f"Current: {dominant_region.value}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['region_text'], 2)
            y_offset += line_height
        
        # Time window dominant regions
        for window_key in ['5s', '10s', '15s']:
            region = dominant_regions_windows.get(window_key)
            if region:
                text = f"{window_key}: {region.value}"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           self.colors['region_text'], 1)
                y_offset += line_height
        
        return frame
    
    def draw_info_overlay(self, frame: np.ndarray, tracking_result: dict) -> np.ndarray:
        """Draw information overlay on frame"""
        # Get FPS
        fps = tracking_result.get('fps', 0.0)
        
        # Prepare info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Face: {'Yes' if tracking_result.get('face_detected') else 'No'}"
        ]
        
        # Draw info in top-right corner
        y_offset = 30
        x_offset = frame.shape[1] - 200
        
        for line in info_lines:
            # Add background for better readability
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, 
                         (x_offset - 5, y_offset - text_height - 5),
                         (x_offset + text_width + 5, y_offset + 5),
                         self.colors['background'], -1)
            
            cv2.putText(frame, line, (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.colors['text'], 1)
            y_offset += 25
        
        return frame
    
    def analyze_depth_data(self, depth_frame: np.ndarray) -> dict:
        """
        Analyze depth data from RealSense camera
        
        Args:
            depth_frame: Depth frame in millimeters (uint16)
            
        Returns:
            dict: Depth statistics in meters
        """
        try:
            # Convert depth from millimeters to meters
            depth_meters = depth_frame.astype(np.float32) / 1000.0
            
            # Filter out zero values (invalid depth)
            valid_depths = depth_meters[depth_meters > 0]
            
            if len(valid_depths) == 0:
                return {
                    'avg_meters': 0.0,
                    'min_meters': 0.0,
                    'max_meters': 0.0,
                    'std_meters': 0.0
                }
            
            # Calculate statistics
            return {
                'avg_meters': float(np.mean(valid_depths)),
                'min_meters': float(np.min(valid_depths)),
                'max_meters': float(np.max(valid_depths)),
                'std_meters': float(np.std(valid_depths))
            }
            
        except Exception as e:
            logger.error(f"Depth analysis error: {e}")
            return {
                'avg_meters': 0.0,
                'min_meters': 0.0,
                'max_meters': 0.0,
                'std_meters': 0.0
            }
    
    def process_frame(self, frame: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """Process a single frame and return visualization"""
        # Track face
        tracking_result = self.tracker.process_frame(frame)
        
        # Add frame number to result
        tracking_result['frame_number'] = self.frame_count
        
        # Add camera type and depth statistics
        tracking_result['camera_type'] = self.camera_manager.get_camera_type()
        
        # Process depth data if available
        if depth is not None:
            depth_stats = self.analyze_depth_data(depth)
            tracking_result['depth_stats'] = depth_stats
        else:
            tracking_result['depth_stats'] = {
                'avg_meters': 0.0,
                'min_meters': 0.0,
                'max_meters': 0.0,
                'std_meters': 0.0
            }
        
        # Log data
        self.logger.log_frame_data(tracking_result)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw face bounding box
        vis_frame = self.draw_face_bounding_box(vis_frame, tracking_result.get('landmarks'))
        
        # Draw landmarks
        vis_frame = self.draw_landmarks(vis_frame, tracking_result.get('landmarks'))
        
        # Apply heatmap overlay
        if self.show_heatmap and tracking_result.get('heatmap') is not None:
            heatmap = tracking_result['heatmap']
            heatmap_normalized = self.tracker.heatmap_generator.normalize_heatmap(heatmap)
            heatmap_colored = self.tracker.heatmap_generator.apply_colormap(heatmap_normalized)
            vis_frame = self.tracker.heatmap_generator.overlay_heatmap(
                vis_frame, heatmap_colored, alpha=0.6
            )
        
        # Apply FaceMap overlay
        if self.config_manager.get('display.show_facemap', True):
            landmarks = self.tracker.get_landmarks_pixels()
            vis_frame = self.facemap.draw_facemap(vis_frame, landmarks)
        
        # Draw region labels
        vis_frame = self.draw_region_labels(vis_frame, tracking_result)
        
        # Draw info overlay
        vis_frame = self.draw_info_overlay(vis_frame, tracking_result)
        
        return vis_frame
    
    def process_frame_heatmap_only(self, frame: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """Process frame with heatmap only visualization"""
        # Track face
        tracking_result = self.tracker.process_frame(frame)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw face bounding box
        vis_frame = self.draw_face_bounding_box(vis_frame, tracking_result.get('landmarks'))
        
        # Draw landmarks (face mesh) like the main window
        vis_frame = self.draw_landmarks(vis_frame, tracking_result.get('landmarks'))
        
        # Apply heatmap overlay
        if tracking_result.get('heatmap') is not None:
            heatmap = tracking_result['heatmap']
            heatmap_normalized = self.tracker.heatmap_generator.normalize_heatmap(heatmap)
            heatmap_colored = self.tracker.heatmap_generator.apply_colormap(heatmap_normalized)
            vis_frame = self.tracker.heatmap_generator.overlay_heatmap(
                vis_frame, heatmap_colored, alpha=0.6
            )
        
        # Add window label
        cv2.putText(vis_frame, "HEATMAP ONLY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame
    
    def process_frame_facemap_only(self, frame: np.ndarray, depth: np.ndarray = None) -> np.ndarray:
        """Process frame with FaceMap only visualization"""
        # Track face
        tracking_result = self.tracker.process_frame(frame)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw face bounding box
        vis_frame = self.draw_face_bounding_box(vis_frame, tracking_result.get('landmarks'))
        
        # Apply FaceMap overlay
        landmarks = self.tracker.get_landmarks_pixels()
        vis_frame = self.facemap.draw_facemap(vis_frame, landmarks)
        
        # Add window label
        cv2.putText(vis_frame, "FACEMAP ONLY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame
    
    def run(self):
        """Main application loop with multi-window visualization"""
        print("Starting Face Heatmap Tracking Application...")
        print("🖥️  MULTI-WINDOW MODE:")
        print("   Window 1: Heatmap + FaceMap (Full Features)")
        print("   Window 2: Heatmap Only")
        print("   Window 3: FaceMap Only")
        print("Press 'q' to quit, 's' to save screenshot, 'h' to toggle heatmap")
        print("Press 'l' to toggle landmarks, 'r' to toggle region labels, 'f' to toggle FaceMap")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        # Create three windows
        window_names = [
            "Face Heatmap + FaceMap",
            "Heatmap Only",
            "FaceMap Only"
        ]
        
        for window_name in window_names:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            self.running = True
            while self.running:
                # Read frame using CameraManager
                frame, depth = self.camera_manager.get_frame()
                if frame is None:
                    print("Error: Could not read frame from camera")
                    break
                
                # Initialize video writer on first frame (save main window only)
                if self.video_writer is None and self.save_video:
                    self.initialize_video_writer(frame.shape)
                
                # Process frame for each visualization
                vis_frame_full = self.process_frame(frame, depth)
                vis_frame_heatmap = self.process_frame_heatmap_only(frame, depth)
                vis_frame_facemap = self.process_frame_facemap_only(frame, depth)
                
                # Resize for display if needed
                if vis_frame_full.shape[1] != self.display_width or vis_frame_full.shape[0] != self.display_height:
                    vis_frame_full = cv2.resize(vis_frame_full, (self.display_width, self.display_height))
                    vis_frame_heatmap = cv2.resize(vis_frame_heatmap, (self.display_width, self.display_height))
                    vis_frame_facemap = cv2.resize(vis_frame_facemap, (self.display_width, self.display_height))
                
                # Show all three windows
                cv2.imshow(window_names[0], vis_frame_full)
                cv2.imshow(window_names[1], vis_frame_heatmap)
                cv2.imshow(window_names[2], vis_frame_facemap)
                
                # Save frame if video recording is enabled (main window only)
                if self.save_video and self.video_writer is not None:
                    self.video_writer.write(vis_frame_full)
                
                # Handle keyboard input (any window can receive input)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self.save_screenshot(vis_frame_full, "full")
                    self.save_screenshot(vis_frame_heatmap, "heatmap")
                    self.save_screenshot(vis_frame_facemap, "facemap")
                elif key == ord('h'):
                    self.show_heatmap = not self.show_heatmap
                    print(f"Heatmap display: {'ON' if self.show_heatmap else 'OFF'}")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('r'):
                    self.show_regions = not self.show_regions
                    print(f"Region labels: {'ON' if self.show_regions else 'OFF'}")
                elif key == ord('f'):
                    self.facemap.toggle()
                    print(f"FaceMap overlay: {'ON' if self.facemap.is_enabled() else 'OFF'}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def save_screenshot(self, frame: np.ndarray, window_type: str = "full"):
        """Save current frame as screenshot with window type"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = os.path.join(self.output_dir, f"screenshot_{window_type}_{timestamp}.jpg")
        cv2.imwrite(screenshot_file, frame)
        print(f"Screenshot saved: {screenshot_file}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        
        # Force write any remaining log data
        self.logger.force_write()
        
        # Generate summary report
        self.logger.export_summary_report()
        
        # Release camera using CameraManager
        if self.camera_manager is not None:
            self.camera_manager.release()
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
        
        # Destroy all windows
        cv2.destroyAllWindows()
        
        print("All windows closed and resources cleaned up")
        
        if self.tracker is not None:
            self.tracker.cleanup()
        
        print(f"Session ended. Total frames processed: {self.frame_count}")
        print(f"Output files saved to: {self.output_dir}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real-time Face Heatmap Tracking System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--save-video", action="store_true", default=True, help="Save video output")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--no-landmarks", action="store_true", help="Hide face landmarks")
    parser.add_argument("--no-heatmap", action="store_true", help="Hide heatmap overlay")
    parser.add_argument("--no-regions", action="store_true", help="Hide region labels")
    
    args = parser.parse_args()
    
    # Create application with video saving enabled by default
    app = FaceHeatmapApp(
        camera_index=args.camera,
        save_video=args.save_video,
        output_dir=args.output_dir,
        show_landmarks=not args.no_landmarks,
        show_heatmap=not args.no_heatmap,
        show_regions=not args.no_regions
    )
    
    # Run application
    app.run()

if __name__ == "__main__":
    main()
