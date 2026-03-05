"""
Configuration Module
Configuration settings for the face heatmap tracking system
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from regions import FacialRegion

@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation"""
    sigma: float = 15.0
    temporal_smoothing: float = 0.3
    buffer_size: int = 5
    alpha: float = 0.6  # Overlay transparency
    colormap: int = 11  # cv2.COLORMAP_JET

@dataclass
class TrackingConfig:
    """Configuration for face tracking"""
    time_windows: List[float] = None
    activity_threshold: float = 0.01
    dominance_debounce_frames: int = 5  # Phase 4 fix: Added for dominance debouncing
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = [5.0, 10.0, 15.0]

@dataclass
class LoggingConfig:
    """Configuration for data logging"""
    output_file: str = "realtime_face_heatmap_log.csv"
    buffer_size: int = 100
    auto_flush_interval: float = 5.0  # seconds
    generate_summary: bool = True

@dataclass
class DisplayConfig:
    """Configuration for display settings"""
    window_name: str = "Face Heatmap Tracking"
    width: int = 1280
    height: int = 720
    show_landmarks: bool = True
    show_heatmap: bool = True
    show_regions: bool = True
    show_facemap: bool = True
    show_fps: bool = True
    show_timestamp: bool = True

@dataclass
class RegionConfig:
    """Configuration for facial regions"""
    enabled_regions: List[str] = None
    region_colors: Dict[str, List[int]] = None
    
    def __post_init__(self):
        if self.enabled_regions is None:
            # Enable all regions except FULL_FACE for individual analysis
            self.enabled_regions = [
                region.value for region in FacialRegion 
                if region != FacialRegion.FULL_FACE
            ]
        
        if self.region_colors is None:
            # Default colors will be used from regions.py
            self.region_colors = {}

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    target_fps: int = 30
    max_frame_size: tuple = (1920, 1080)
    enable_gpu: bool = False
    processing_threads: int = 1

@dataclass
class FaceMapConfig:
    """Configuration for FaceMap overlay system"""
    enabled: bool = True
    color: List[int] = None  # BGR format
    thickness: int = 2
    show_labels: bool = True
    font_scale: float = 0.25
    
    def __post_init__(self):
        if self.color is None:
            self.color = [0, 255, 0]  # Green (BGR format)

@dataclass
class AppConfig:
    """Main application configuration"""
    heatmap: HeatmapConfig = None
    tracking: TrackingConfig = None
    logging: LoggingConfig = None
    display: DisplayConfig = None
    regions: RegionConfig = None
    performance: PerformanceConfig = None
    facemap: FaceMapConfig = None
    camera_index: int = 0
    save_video: bool = False
    output_dir: str = "output"
    
    def __post_init__(self):
        if self.heatmap is None:
            self.heatmap = HeatmapConfig()
        if self.tracking is None:
            self.tracking = TrackingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.display is None:
            self.display = DisplayConfig()
        if self.regions is None:
            self.regions = RegionConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.facemap is None:
            self.facemap = FaceMapConfig()

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = AppConfig()
        # Load config on initialization
        self.load_config()
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation (e.g., 'facemap.enabled')
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except (AttributeError, KeyError):
            return default
    
    def load_config(self) -> AppConfig:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update config with loaded data
            if 'heatmap' in config_data:
                self.config.heatmap = HeatmapConfig(**config_data['heatmap'])
            
            if 'tracking' in config_data:
                self.config.tracking = TrackingConfig(**config_data['tracking'])
            
            if 'logging' in config_data:
                self.config.logging = LoggingConfig(**config_data['logging'])
            
            if 'display' in config_data:
                self.config.display = DisplayConfig(**config_data['display'])
            
            if 'regions' in config_data:
                self.config.regions = RegionConfig(**config_data['regions'])
            
            if 'performance' in config_data:
                self.config.performance = PerformanceConfig(**config_data['performance'])
            
            if 'facemap' in config_data:
                self.config.facemap = FaceMapConfig(**config_data['facemap'])
            
            # Update main config fields
            for key in ['camera_index', 'save_video', 'output_dir']:
                if key in config_data:
                    setattr(self.config, key, config_data[key])
            
            print(f"Configuration loaded from {self.config_file}")
            
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found, using defaults")
            self.save_config()  # Save default config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
        
        return self.config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config key: {key}")
    
    def get_region_config(self, region_name: str) -> Dict[str, Any]:
        """Get configuration for a specific region"""
        return {
            'enabled': region_name in self.config.regions.enabled_regions,
            'color': self.config.regions.region_colors.get(region_name)
        }
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        valid = True
        
        # Validate heatmap config
        if self.config.heatmap.sigma <= 0:
            print("Error: Heatmap sigma must be positive")
            valid = False
        
        if not 0 <= self.config.heatmap.temporal_smoothing <= 1:
            print("Error: Temporal smoothing must be between 0 and 1")
            valid = False
        
        # Validate tracking config
        if any(window <= 0 for window in self.config.tracking.time_windows):
            print("Error: Time windows must be positive")
            valid = False
        
        if not 0 <= self.config.tracking.min_detection_confidence <= 1:
            print("Error: Detection confidence must be between 0 and 1")
            valid = False
        
        # Validate display config
        if self.config.display.width <= 0 or self.config.display.height <= 0:
            print("Error: Display dimensions must be positive")
            valid = False
        
        # Validate performance config
        if self.config.performance.target_fps <= 0:
            print("Error: Target FPS must be positive")
            valid = False
        
        return valid

# Global config instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the current application configuration"""
    return config_manager.config

def load_config(config_file: str = "config.json") -> AppConfig:
    """Load configuration from file"""
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager.load_config()

def save_config():
    """Save current configuration to file"""
    config_manager.save_config()
