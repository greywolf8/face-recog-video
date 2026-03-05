"""
Face Tracking Module
Main face tracking logic with region activity detection and temporal analysis
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import time
import json

from regions import FacialRegions, FacialRegion
from heatmap import HeatmapGenerator

class FaceTracker:
    """Main face tracking class with region activity detection"""
    
    def __init__(self, time_windows: List[float] = [5.0, 10.0, 15.0]):
        """
        Initialize face tracker
        
        Args:
            time_windows: List of time windows in seconds for activity analysis
        """
        # Initialize MediaPipe Face Mesh with performance optimizations
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Disable refinement for better performance
            min_detection_confidence=0.3,  # Lower confidence for faster detection
            min_tracking_confidence=0.3,   # Lower tracking confidence
            static_image_mode=False        # Enable video mode for better performance
        )
        
        # Initialize regions and heatmap generator
        self.regions = FacialRegions()
        self.heatmap_generator = None
        
        # Time windows for activity analysis
        self.time_windows = time_windows
        
        # Activity tracking data structures
        self.region_activity_history = defaultdict(lambda: deque(maxlen=1000))
        self.dominant_region_history = deque(maxlen=1000)
        
        # Frame tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Previous landmarks for movement calculation
        self.prev_landmarks = None
        
        # Current frame data
        self.current_landmarks = None
        self.current_activity_scores = {}
        self.current_dominant_region = None
        self.heatmap_stats = {}
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
    
    def initialize_heatmap_generator(self, frame_shape: Tuple[int, int]):
        """Initialize heatmap generator with frame dimensions"""
        if self.heatmap_generator is None:
            height, width = frame_shape
            self.heatmap_generator = HeatmapGenerator(width, height)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and extract face landmarks
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary containing tracking results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize result dictionary
        result = {
            'face_detected': False,
            'landmarks': None,
            'activity_scores': {},
            'dominant_region': None,
            'dominant_regions_windows': {},
            'heatmap': None,
            'fps': 0.0,
            'timestamp': time.time()
        }
        
        if results.multi_face_landmarks:
            # Extract landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array (x, y coordinates normalized to image size)
            h, w = frame.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            self.current_landmarks = np.array(landmarks)
            result['landmarks'] = self.current_landmarks
            result['face_detected'] = True
            
            # Initialize heatmap generator if needed
            self.initialize_heatmap_generator(frame.shape[:2])
            
            # Calculate activity scores
            self._calculate_activity_scores()
            result['activity_scores'] = self.current_activity_scores.copy()
            
            # Find dominant region
            self._find_dominant_region()
            result['dominant_region'] = self.current_dominant_region
            
            # Calculate dominant regions for time windows
            result['dominant_regions_windows'] = self._get_dominant_regions_windows()
            
            # Generate heatmap
            result['heatmap'] = self._generate_heatmap()
            
            # Calculate heatmap statistics
            if result['heatmap'] is not None:
                self.heatmap_stats = self.heatmap_generator.calculate_heatmap_stats(result['heatmap'])
                result['heatmap_stats'] = self.heatmap_stats
        
        # Update frame tracking
        self._update_frame_tracking()
        result['fps'] = self._calculate_fps()
        
        # Store previous landmarks
        if self.current_landmarks is not None:
            self.prev_landmarks = self.current_landmarks.copy()
        
        self.frame_count += 1
        
        return result
    
    def get_landmarks_pixels(self) -> Optional[np.ndarray]:
        """
        Get current face landmarks in pixel coordinates
        
        Returns:
            numpy array of shape (468, 2) containing (x, y) pixel coordinates,
            or None if no face is currently detected
        """
        return self.current_landmarks.copy() if self.current_landmarks is not None else None
    
    def _calculate_activity_scores(self):
        """Calculate activity scores for all facial regions"""
        if self.current_landmarks is None:
            return
        
        current_time = time.time()
        
        for region in FacialRegion:
            if region == FacialRegion.FULL_FACE:
                continue  # Skip full face for individual region analysis
            
            # Calculate activity score
            if self.prev_landmarks is not None:
                activity_score = self.regions.calculate_region_activity_score(
                    self.current_landmarks, self.prev_landmarks, region
                )
            else:
                activity_score = 0.0
            
            # Store current activity score
            self.current_activity_scores[region] = activity_score
            
            # Add to history with timestamp
            self.region_activity_history[region].append({
                'timestamp': current_time,
                'score': activity_score
            })
    
    def _find_dominant_region(self):
        """Find the most active region in current frame"""
        if not self.current_activity_scores:
            self.current_dominant_region = None
            return
        
        # Find region with highest activity score
        dominant_region = max(self.current_activity_scores.items(), key=lambda x: x[1])
        
        # Only consider it dominant if activity is above threshold
        if dominant_region[1] > 0.01:  # Threshold to avoid noise
            self.current_dominant_region = dominant_region[0]
        else:
            self.current_dominant_region = None
        
        # Add to history
        current_time = time.time()
        self.dominant_region_history.append({
            'timestamp': current_time,
            'region': self.current_dominant_region
        })
    
    def _get_dominant_regions_windows(self) -> Dict[str, Optional[FacialRegion]]:
        """Calculate dominant regions for different time windows"""
        current_time = time.time()
        window_results = {}
        
        for window_duration in self.time_windows:
            window_start = current_time - window_duration
            
            # Filter dominant region history to time window
            window_regions = []
            for entry in self.dominant_region_history:
                if entry['timestamp'] >= window_start and entry['region'] is not None:
                    window_regions.append(entry['region'])
            
            # Find most frequent region in window
            if window_regions:
                from collections import Counter
                region_counts = Counter(window_regions)
                most_common_region = region_counts.most_common(1)[0][0]
            else:
                most_common_region = None
            
            window_results[f'{int(window_duration)}s'] = most_common_region
        
        return window_results
    
    def _generate_heatmap(self) -> Optional[np.ndarray]:
        """Generate heatmap based on current landmarks"""
        if self.current_landmarks is None or self.heatmap_generator is None:
            return None
        
        # Calculate movement weights
        movement_weights = None
        if self.prev_landmarks is not None:
            movement = np.linalg.norm(self.current_landmarks - self.prev_landmarks, axis=1)
            movement_weights = movement / (np.max(movement) + 1e-6)
        
        # Generate heatmap
        heatmap = self.heatmap_generator.generate_heatmap(
            self.current_landmarks, movement_weights
        )
        
        return heatmap
    
    def _update_frame_tracking(self):
        """Update frame tracking metrics"""
        self.last_frame_time = time.time()
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        
        if len(self.fps_history) > 0:
            time_diff = current_time - self.fps_history[-1]
            if time_diff > 0:
                fps = 1.0 / time_diff
            else:
                fps = 0.0
        else:
            fps = 0.0
        
        self.fps_history.append(current_time)
        
        # Return average FPS over recent frames
        if len(self.fps_history) > 1:
            time_span = self.fps_history[-1] - self.fps_history[0]
            if time_span > 0:
                avg_fps = (len(self.fps_history) - 1) / time_span
            else:
                avg_fps = fps
        else:
            avg_fps = fps
        
        return avg_fps
    
    def get_region_activity_json(self) -> str:
        """Get current activity scores as JSON string"""
        return json.dumps({
            region.value: score 
            for region, score in self.current_activity_scores.items()
        })
    
    def get_face_bounding_box(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box around face landmarks
        
        Args:
            landmarks: Face landmark coordinates
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if landmarks is None or len(landmarks) == 0:
            return (0, 0, 0, 0)
        
        x_min = int(np.min(landmarks[:, 0]))
        y_min = int(np.min(landmarks[:, 1]))
        x_max = int(np.max(landmarks[:, 0]))
        y_max = int(np.max(landmarks[:, 1]))
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding
        
        return (x_min, y_min, x_max, y_max)
    
    def get_landmarks_by_region(self, region: FacialRegion) -> np.ndarray:
        """Get landmarks for a specific region"""
        if self.current_landmarks is None:
            return np.array([])
        
        return self.regions.filter_landmarks_by_region(self.current_landmarks, region)
    
    def get_region_centroid(self, region: FacialRegion) -> np.ndarray:
        """Get centroid of a specific region"""
        if self.current_landmarks is None:
            return np.array([0, 0])
        
        return self.regions.calculate_region_centroid(self.current_landmarks, region)
    
    def reset(self):
        """Reset tracker state"""
        self.region_activity_history.clear()
        self.dominant_region_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.prev_landmarks = None
        self.current_landmarks = None
        self.current_activity_scores.clear()
        self.current_dominant_region = None
        self.heatmap_stats.clear()
        self.fps_history.clear()
        
        if self.heatmap_generator:
            self.heatmap_generator.reset()
    
    def cleanup(self):
        """Clean up resources"""
        if self.face_mesh and hasattr(self.face_mesh, '_graph') and self.face_mesh._graph is not None:
            self.face_mesh.close()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction
