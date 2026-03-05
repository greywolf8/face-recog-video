"""
Facial Regions Definition Module
Defines landmark index groups for different facial regions using MediaPipe Face Mesh (468 landmarks)
"""

import numpy as np
from typing import Dict, List, Set
from enum import Enum

class FacialRegion(Enum):
    """Enum for facial regions"""
    RIGHT_EYEBROW = "RIGHT_EYEBROW"
    LEFT_EYEBROW = "LEFT_EYEBROW"
    RIGHT_EYE = "RIGHT_EYE"
    LEFT_EYE = "LEFT_EYE"
    NOSE = "NOSE"
    LIPS = "LIPS"
    FOREHEAD = "FOREHEAD"
    CHIN = "CHIN"
    RIGHT_CHEEK = "RIGHT_CHEEK"
    LEFT_CHEEK = "LEFT_CHEEK"
    RIGHT_EAR = "RIGHT_EAR"
    LEFT_EAR = "LEFT_EAR"
    HAIR = "HAIR"
    NECK = "NECK"
    RIGHT_SHOULDER = "RIGHT_SHOULDER"
    LEFT_SHOULDER = "LEFT_SHOULDER"
    FULL_FACE = "FULL_FACE"

class FacialRegions:
    """Class to manage facial region landmark mappings"""
    
    def __init__(self):
        self.regions = self._define_regions()
    
    def _define_regions(self) -> Dict[FacialRegion, List[int]]:
        """
        Define landmark index groups for each facial region based on MediaPipe Face Mesh topology
        Returns dictionary mapping region names to landmark indices
        """
        regions = {
            # Right eyebrow landmarks (indices 46-70)
            FacialRegion.RIGHT_EYEBROW: list(range(46, 71)),
            
            # Left eyebrow landmarks (indices 70-94, but 70 is shared)
            FacialRegion.LEFT_EYEBROW: list(range(70, 95)),
            
            # Right eye landmarks (indices 33-133, but using specific eye contour)
            FacialRegion.RIGHT_EYE: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            
            # Left eye landmarks (indices 362-398, but using specific eye contour)
            FacialRegion.LEFT_EYE: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            
            # Nose landmarks (indices 1-68, 275-295, 331-361)
            FacialRegion.NOSE: list(range(1, 69)) + list(range(275, 296)) + list(range(331, 362)),
            
            # Lips landmarks (indices 0-17, 61-80, 84-87, 91-146, 178-191, 267-270, 274-285, 308-324, 375-402)
            FacialRegion.LIPS: list(range(0, 18)) + list(range(61, 81)) + list(range(84, 88)) + 
                           list(range(91, 147)) + list(range(178, 192)) + list(range(267, 271)) + 
                           list(range(274, 286)) + list(range(308, 325)) + list(range(375, 403)),
            
            # Forehead landmarks (upper face region)
            FacialRegion.FOREHEAD: list(range(10, 35)) + list(range(65, 70)) + list(range(105, 110)) + 
                                 list(range(151, 156)) + list(range(286, 295)) + list(range(334, 339)) + 
                                 list(range(296, 301)) + list(range(336, 341)),
            
            # Chin and jawline landmarks (indices 172-187, 401-415)
            FacialRegion.CHIN: list(range(172, 188)) + list(range(401, 416)) + list(range(148, 151)) + 
                             list(range(176, 179)) + list(range(397, 400)) + list(range(425, 430)),
            
            # Right cheek landmarks
            FacialRegion.RIGHT_CHEEK: list(range(50, 65)) + list(range(116, 131)) + list(range(205, 220)) + 
                                   list(range(280, 295)) + list(range(340, 355)),
            
            # Left cheek landmarks
            FacialRegion.LEFT_CHEEK: list(range(280, 295)) + list(range(340, 355)) + list(range(345, 360)) + 
                                  list(range(410, 425)) + list(range(445, 460)),
            
            # Right ear landmarks (approximate using side face landmarks)
            FacialRegion.RIGHT_EAR: list(range(45, 54)) + list(range(234, 247)) + list(range(127, 140)) + 
                                  list(range(162, 175)),
            
            # Left ear landmarks (approximate using side face landmarks)
            FacialRegion.LEFT_EAR: list(range(274, 285)) + list(range(354, 367)) + list(range(456, 468)) + 
                                 list(range(423, 436)),
            
            # Hair region (upper face boundary approximation)
            FacialRegion.HAIR: list(range(10, 35)) + list(range(65, 70)) + list(range(105, 110)) + 
                             list(range(151, 156)) + list(range(286, 295)) + list(range(334, 339)),
            
            # Neck region (lower jaw extension approximation)
            FacialRegion.NECK: list(range(172, 188)) + list(range(401, 416)) + list(range(148, 151)) + 
                             list(range(176, 179)) + list(range(397, 400)),
            
            # Right shoulder (approximate edge of frame when pose is detected)
            FacialRegion.RIGHT_SHOULDER: [],  # Will be populated if pose landmarks are available
            
            # Left shoulder (approximate edge of frame when pose is detected)
            FacialRegion.LEFT_SHOULDER: [],   # Will be populated if pose landmarks are available
            
            # Full face (all landmarks)
            FacialRegion.FULL_FACE: list(range(468))
        }
        
        return regions
    
    def get_region_landmarks(self, region: FacialRegion) -> List[int]:
        """Get landmark indices for a specific region"""
        return self.regions.get(region, [])
    
    def get_region_name(self, region: FacialRegion) -> str:
        """Get the string name of a region"""
        return region.value
    
    def get_all_regions(self) -> Dict[FacialRegion, List[int]]:
        """Get all region mappings"""
        return self.regions
    
    def get_landmarks_for_regions(self, regions: List[FacialRegion]) -> Set[int]:
        """Get unique landmark indices for multiple regions"""
        landmarks = set()
        for region in regions:
            landmarks.update(self.get_region_landmarks(region))
        return landmarks
    
    def filter_landmarks_by_region(self, landmarks: np.ndarray, region: FacialRegion) -> np.ndarray:
        """
        Filter landmarks array to only include landmarks from specified region
        landmarks: numpy array of shape (468, 2) or (468, 3)
        """
        region_indices = self.get_region_landmarks(region)
        if len(region_indices) == 0:
            return np.array([])
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in region_indices if idx < len(landmarks)]
        
        if len(valid_indices) == 0:
            return np.array([])
        
        return landmarks[valid_indices]
    
    def calculate_region_centroid(self, landmarks: np.ndarray, region: FacialRegion) -> np.ndarray:
        """
        Calculate centroid of landmarks in a region
        Returns np.array([x, y]) or np.array([x, y, z]) depending on input
        """
        region_landmarks = self.filter_landmarks_by_region(landmarks, region)
        
        if len(region_landmarks) == 0:
            return np.array([0, 0]) if landmarks.shape[1] == 2 else np.array([0, 0, 0])
        
        return np.mean(region_landmarks, axis=0)
    
    def calculate_region_activity_score(self, landmarks: np.ndarray, prev_landmarks: np.ndarray, 
                                      region: FacialRegion) -> float:
        """
        Calculate activity score for a region based on movement and landmark density
        with improved normalization (Phase 5 fix)
        landmarks: current frame landmarks
        prev_landmarks: previous frame landmarks
        """
        region_landmarks = self.filter_landmarks_by_region(landmarks, region)
        prev_region_landmarks = self.filter_landmarks_by_region(prev_landmarks, region)
        
        if len(region_landmarks) == 0 or len(prev_region_landmarks) == 0:
            return 0.0
        
        # Calculate movement magnitude
        if len(region_landmarks) == len(prev_region_landmarks):
            movement = np.linalg.norm(region_landmarks - prev_region_landmarks, axis=1)
            avg_movement = np.mean(movement)
        else:
            avg_movement = 0.0
        
        # Phase 5 fix: Improved normalization to avoid over-normalization
        # 1) Normalize relative to global max movement of frame (not per-region)
        global_movement = np.linalg.norm(landmarks - prev_landmarks, axis=1)
        global_max_movement = np.max(global_movement)
        
        if global_max_movement > 0:
            normalized_movement = avg_movement / global_max_movement
        else:
            normalized_movement = 0.0
        
        # 2) Apply area-weight normalization to penalize large regions
        region_area = len(region_landmarks)
        area_penalty = 1.0 / (1.0 + region_area / 100.0)  # Penalize regions with many landmarks
        
        # 3) Apply softmax-like temperature control for smoothing
        temperature = 2.0  # Higher temperature = more smoothing
        softmax_score = np.exp(normalized_movement / temperature)
        
        # 4) Combined activity score with improved normalization
        activity_score = softmax_score * area_penalty
        
        return float(activity_score)
    
    def get_region_colors(self) -> Dict[FacialRegion, tuple]:
        """Get colors for visualizing different regions"""
        colors = {
            FacialRegion.RIGHT_EYEBROW: (255, 0, 0),      # Red
            FacialRegion.LEFT_EYEBROW: (0, 255, 0),       # Green
            FacialRegion.RIGHT_EYE: (0, 0, 255),           # Blue
            FacialRegion.LEFT_EYE: (255, 255, 0),          # Cyan
            FacialRegion.NOSE: (255, 0, 255),              # Magenta
            FacialRegion.LIPS: (0, 255, 255),              # Yellow
            FacialRegion.FOREHEAD: (128, 0, 128),          # Purple
            FacialRegion.CHIN: (255, 165, 0),              # Orange
            FacialRegion.RIGHT_CHEEK: (255, 192, 203),     # Pink
            FacialRegion.LEFT_CHEEK: (165, 42, 42),        # Brown
            FacialRegion.RIGHT_EAR: (128, 128, 128),       # Gray
            FacialRegion.LEFT_EAR: (192, 192, 192),        # Light Gray
            FacialRegion.HAIR: (0, 0, 0),                  # Black
            FacialRegion.NECK: (64, 64, 64),               # Dark Gray
            FacialRegion.RIGHT_SHOULDER: (100, 149, 237),  # Cornflower Blue
            FacialRegion.LEFT_SHOULDER: (255, 215, 0),     # Gold
            FacialRegion.FULL_FACE: (255, 255, 255)        # White
        }
        return colors

# Utility function to get region by name
def get_region_by_name(name: str) -> FacialRegion:
    """Get FacialRegion enum by string name"""
    for region in FacialRegion:
        if region.value == name:
            return region
    raise ValueError(f"Region '{name}' not found")
