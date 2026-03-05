"""
Camera Manager - Unified camera abstraction for webcam and Intel RealSense

This module provides a unified interface for both PC webcam and Intel RealSense cameras,
maintaining compatibility with the existing face tracking pipeline.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging

# Try to import RealSense - handle gracefully if not available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Unified camera manager supporting both webcam and Intel RealSense cameras.
    
    Provides consistent BGR frame output regardless of camera source.
    """
    
    def __init__(self):
        self.camera_type = None  # 'webcam' or 'realsense'
        self.webcam = None
        self.pipeline = None
        self.config = None
        self.align = None
        self.is_initialized = False
        
    def select_camera(self) -> str:
        """
        Prompt user to select camera source.
        
        Returns:
            str: Selected camera type ('webcam' or 'realsense')
        """
        print("\n" + "="*50)
        print("Select Camera Source:")
        print("1 - PC / Laptop Webcam (Default)")
        print("2 - Intel RealSense Camera (RGB + Depth)")
        print("="*50)
        
        while True:
            try:
                choice = input("Enter choice (1 or 2): ").strip()
                if choice == '1':
                    return 'webcam'
                elif choice == '2':
                    return 'realsense'
                elif choice == '':
                    return 'webcam'  # Default
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except (EOFError, KeyboardInterrupt):
                print("\nUsing webcam as default.")
                return 'webcam'
    
    def initialize(self, camera_type: str = None, webcam_index: int = 0) -> bool:
        """
        Initialize the selected camera.
        
        Args:
            camera_type: Type of camera ('webcam' or 'realsense')
            webcam_index: Index for webcam (default: 0)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if camera_type is None:
            camera_type = self.select_camera()
        
        self.camera_type = camera_type
        
        if camera_type == 'webcam':
            return self._initialize_webcam(webcam_index)
        elif camera_type == 'realsense':
            return self._initialize_realsense()
        else:
            logger.error(f"Unknown camera type: {camera_type}")
            return False
    
    def _initialize_webcam(self, index: int) -> bool:
        """Initialize webcam camera."""
        try:
            self.webcam = cv2.VideoCapture(index)
            if not self.webcam.isOpened():
                logger.error(f"Failed to open webcam at index {index}")
                return False
            
            # Set resolution for performance (matching existing system)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_initialized = True
            logger.info(f"Webcam initialized at index {index}")
            return True
            
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            return False
    
    def _initialize_realsense(self) -> bool:
        """Initialize Intel RealSense camera."""
        if not REALSENSE_AVAILABLE:
            logger.error("RealSense module not found. Install: pip install pyrealsense2")
            print("Warning: RealSense module not found. Falling back to webcam.")
            return self._initialize_webcam(0)
        
        try:
            # Create RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams (matching existing system resolution)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Create align object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            self.is_initialized = True
            logger.info("RealSense camera initialized")
            return True
            
        except Exception as e:
            logger.error(f"RealSense initialization failed: {e}")
            print("Warning: RealSense camera not found or failed to initialize. Falling back to webcam.")
            return self._initialize_webcam(0)
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get frame from camera.
        
        Returns:
            Tuple: (frame_bgr, depth_frame_or_none)
                   Returns (None, None) if frame capture fails
        """
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return None, None
        
        if self.camera_type == 'webcam':
            return self._get_webcam_frame()
        elif self.camera_type == 'realsense':
            return self._get_realsense_frame()
        else:
            logger.error(f"Unknown camera type: {self.camera_type}")
            return None, None
    
    def _get_webcam_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get frame from webcam."""
        try:
            ret, frame = self.webcam.read()
            if ret and frame is not None:
                # Ensure BGR format (OpenCV default)
                return frame, None
            else:
                logger.warning("Failed to read webcam frame")
                return None, None
        except Exception as e:
            logger.error(f"Webcam frame capture error: {e}")
            return None, None
    
    def _get_realsense_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get frame from RealSense camera."""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame:
                logger.warning("No color frame received")
                return None, None
            
            # Convert color frame to numpy array (BGR format)
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert depth frame to numpy array if available
            depth_image = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            logger.error(f"RealSense frame capture error: {e}")
            return None, None
    
    def get_camera_type(self) -> str:
        """Get current camera type."""
        return self.camera_type
    
    def release(self):
        """Release camera resources."""
        try:
            if self.webcam is not None:
                self.webcam.release()
                self.webcam = None
                logger.info("Webcam released")
            
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
                logger.info("RealSense pipeline stopped")
            
            self.is_initialized = False
            
        except Exception as e:
            logger.error(f"Error releasing camera resources: {e}")
    
    def __del__(self):
        """Destructor - ensure resources are released."""
        self.release()


# Utility function for backward compatibility
def create_camera_manager() -> CameraManager:
    """
    Create and return a CameraManager instance.
    
    Returns:
        CameraManager: Configured camera manager instance
    """
    return CameraManager()


# Test function for standalone camera testing
def test_camera_manager():
    """Test camera manager functionality."""
    print("Testing Camera Manager...")
    
    manager = CameraManager()
    
    # Test initialization
    if manager.initialize():
        print(f"Camera initialized: {manager.get_camera_type()}")
        
        # Test frame capture
        for i in range(5):
            frame, depth = manager.get_frame()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}, Depth: {'Yes' if depth is not None else 'No'}")
            else:
                print(f"Frame {i+1}: Failed to capture")
        
        # Test release
        manager.release()
        print("Camera released successfully")
    else:
        print("Camera initialization failed")


if __name__ == "__main__":
    test_camera_manager()
