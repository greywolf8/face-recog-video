"""
FaceMap Overlay Module
Adds anatomical facial region polygons overlay to the existing face tracking system.
Integrates seamlessly with the existing heatmap tracking pipeline.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import config


class FaceMapOverlay:
    """
    FaceMap overlay system that draws anatomical facial regions on detected faces.
    Uses MediaPipe 468 landmarks to generate dynamic, perfectly aligned polygons.
    """
    
    def __init__(self, config_manager: config.ConfigManager):
        """
        Initialize FaceMap overlay system
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.enabled = self.config.get('facemap.enabled', True)
        self.color = tuple(self.config.get('facemap.color', [0, 255, 0]))  # Green
        self.thickness = self.config.get('facemap.thickness', 2)
        self.show_labels = self.config.get('facemap.show_labels', True)
        self.font_scale = self.config.get('facemap.font_scale', 0.25)
        
        # Pre-compute landmark indices for each region (performance optimization)
        self._build_region_mappings()
        
        # Cache for computed polygons to avoid recalculating every frame
        self._polygon_cache = {}
        self._last_landmarks_hash = None
    
    def _build_region_mappings(self) -> None:
        """
        Pre-compute MediaPipe landmark indices for each facial region.
        Uses official MediaPipe Face Mesh topology for accurate anatomical mapping.
        """
        # MediaPipe Face Mesh landmark indices (468 total)
        # These indices correspond to specific facial features
        
        # Right eye landmarks (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        self.region_mappings = {
            'RIGHT_EYE': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            
            # Left eye landmarks (362, 398, 384, 385, 386, 387, 388, 389, 390, 374, 380, 381, 382, 362, 263, 466)
            'LEFT_EYE': [362, 398, 384, 385, 386, 387, 388, 389, 390, 374, 380, 381, 382, 362, 263, 466],
            
            # Right eyebrow (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
            'RIGHT_EYEBROW': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            
            # Left eyebrow (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
            'LEFT_EYEBROW': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            
            # Nose tip and bridge (1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 131, 134, 102, 49, 220, 305, 292, 33, 133, 157, 158, 159, 160, 161, 246)
            'NOSE': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 131, 134, 102, 49, 220, 305, 292, 33, 133, 157, 158, 159, 160, 161, 246],
            
            # Lips outer boundary (61, 84, 17, 314, 405, 291, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
            'LIPS': [61, 84, 17, 314, 405, 291, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            
            # Forehead (upper face mesh: 69, 70, 107, 108, 109, 151, 9, 10, 151, 105, 66, 107, 55, 65, 52, 53, 46)
            'FOREHEAD': [69, 70, 107, 108, 109, 151, 9, 10, 151, 105, 66, 107, 55, 65, 52, 53, 46],
            
            # Chin and jawline (172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464)
            'CHIN': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464],
            
            # Right cheek (approximation using face mesh)
            'RIGHT_CHEEK': [50, 101, 117, 118, 119, 120, 121, 128, 129, 205, 206, 207, 31, 228, 229, 230, 231],
            
            # Left cheek (approximation using face mesh)
            'LEFT_CHEEK': [280, 331, 347, 348, 349, 350, 351, 358, 359, 425, 426, 427, 261, 448, 449, 450, 451],
            
            # Right ear approximation (using face contour)
            'RIGHT_EAR': [234, 127, 162, 21, 54, 103, 67, 109, 10],
            
            # Left ear approximation (using face contour)
            'LEFT_EAR': [454, 356, 389, 251, 284, 333, 297, 339, 240],
            
            # Hairline approximation (upper boundary)
            'HAIR': [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464],
            
            # Neck region (jaw extension)
            'NECK': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 340],
            
            # Full face (convex hull of all landmarks)
            'FULL_FACE': []  # Will be computed dynamically
        }
    
    def _compute_polygon(self, landmarks: np.ndarray, region_name: str) -> np.ndarray:
        """
        Compute polygon points for a specific facial region
        
        Args:
            landmarks: Array of shape (468, 2) containing pixel coordinates
            region_name: Name of the facial region
            
        Returns:
            Array of polygon points in correct order for cv2.polylines
        """
        if region_name == 'FULL_FACE':
            # Compute convex hull of all landmarks for full face boundary
            if len(landmarks) == 0:
                return np.array([])
            
            hull = cv2.convexHull(landmarks.astype(np.float32))
            return hull.astype(np.int32)
        
        indices = self.region_mappings[region_name]
        if not indices or len(landmarks) == 0:
            return np.array([])
        
        # Get landmark coordinates for this region
        region_points = landmarks[indices]
        
        # Order points for proper polygon drawing (convex hull for smooth boundaries)
        if len(region_points) > 2:
            hull = cv2.convexHull(region_points.astype(np.float32))
            return hull.astype(np.int32)
        else:
            return region_points.astype(np.int32)
    
    def _compute_region_centroid(self, polygon: np.ndarray) -> Tuple[int, int]:
        """
        Compute centroid of a polygon for label positioning
        
        Args:
            polygon: Array of polygon points
            
        Returns:
            (x, y) centroid coordinates
        """
        if len(polygon) == 0:
            return (0, 0)
        
        # Flatten polygon and compute mean
        if len(polygon.shape) == 3:
            points = polygon.reshape(-1, 2)
        else:
            points = polygon
        
        centroid_x = int(np.mean(points[:, 0]))
        centroid_y = int(np.mean(points[:, 1]))
        
        return (centroid_x, centroid_y)
    
    def draw_facemap(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw FaceMap overlay on the frame
        
        Args:
            frame: Input BGR image frame
            landmarks: Array of shape (468, 2) containing pixel coordinates
            
        Returns:
            Frame with FaceMap overlay drawn
        """
        if not self.enabled or landmarks is None or len(landmarks) == 0:
            return frame
        
        # Check if we can use cached polygons
        landmarks_hash = hash(landmarks.tobytes())
        if landmarks_hash == self._last_landmarks_hash:
            polygons = self._polygon_cache
        else:
            # Compute new polygons
            polygons = {}
            for region_name in self.region_mappings:
                polygon = self._compute_polygon(landmarks, region_name)
                if len(polygon) > 0:
                    polygons[region_name] = polygon
            
            # Cache for next frame
            self._polygon_cache = polygons
            self._last_landmarks_hash = landmarks_hash
        
        # Draw each region
        for region_name, polygon in polygons.items():
            if len(polygon) == 0:
                continue
                
            # Ensure polygon is in correct format for cv2.polylines
            if len(polygon.shape) == 2:
                # Reshape to (n, 1, 2) format for cv2.polylines
                polygon = polygon.reshape(-1, 1, 2)
            
            # Draw polygon outline
            cv2.polylines(
                frame,
                [polygon],
                isClosed=True,
                color=self.color,
                thickness=self.thickness,
                lineType=cv2.LINE_AA
            )
            
            # Draw region label if enabled
            if self.show_labels:
                centroid = self._compute_region_centroid(polygon)
                if centroid != (0, 0):
                    # Format region name for display (shorter labels)
                    label = region_name.replace('_', ' ').title()
                    # Create shorter labels to reduce space
                    label_map = {
                        'Right Eyebrow': 'R.Brow',
                        'Left Eyebrow': 'L.Brow',
                        'Right Eye': 'R.Eye',
                        'Left Eye': 'L.Eye',
                        'Right Cheek': 'R.Cheek',
                        'Left Cheek': 'L.Cheek',
                        'Right Ear': 'R.Ear',
                        'Left Ear': 'L.Ear',
                        'Right Shoulder': 'R.Shldr',
                        'Left Shoulder': 'L.Shldr',
                        'Forehead': 'Forehd',
                        'Full Face': 'Face'
                    }
                    label = label_map.get(label, label)
                    
                    # Draw text with background for better visibility
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = self.font_scale  # Use configurable font scale
                    thickness = 1
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Draw background rectangle with smaller padding
                    cv2.rectangle(
                        frame,
                        (centroid[0] - text_width // 2 - 1, centroid[1] - text_height // 2 - 1),
                        (centroid[0] + text_width // 2 + 1, centroid[1] + text_height // 2 + baseline + 1),
                        self.color,
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        label,
                        (centroid[0] - text_width // 2, centroid[1] + text_height // 2),
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text for contrast
                        thickness,
                        cv2.LINE_AA
                    )
        
        return frame
    
    def toggle(self) -> None:
        """Toggle FaceMap overlay on/off"""
        self.enabled = not self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable FaceMap overlay"""
        self.enabled = enabled
    
    def is_enabled(self) -> bool:
        """Check if FaceMap overlay is enabled"""
        return self.enabled
