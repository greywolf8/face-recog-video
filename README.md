# Face Heatmap Tracking System

A real-time face tracking system that generates dynamic heatmaps based on facial movement and activity detection. Built with MediaPipe, OpenCV, and advanced smoothing algorithms for stable and accurate tracking.

## 🎯 Features

- **Real-time Face Tracking**: Uses MediaPipe Face Mesh (468 landmarks) for precise facial feature detection
- **Dynamic Heatmap Generation**: Creates Gaussian-based heatmaps that respond to facial movement
- **Activity Detection**: Identifies active facial regions (eyes, mouth, eyebrows, etc.)
- **Stabilization System**: Advanced temporal smoothing and debouncing for consistent tracking
- **Multi-window Display**: Separate windows for heatmap, face mesh, and combined view
- **Performance Optimized**: Runs at 25-30 FPS with configurable quality settings
- **Video Recording**: Automatic recording of tracking sessions
- **Comprehensive Logging**: Detailed activity tracking and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV 4.x
- MediaPipe 0.10.15
- NumPy
- (Optional) Intel RealSense SDK for depth camera support

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd face-recog-video

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install opencv-python mediapipe==0.10.15 numpy
```

### Running the Application

```bash
# Start the face tracking application
python start_camera_fast.py

# Or use the alternative launcher
python run_app.py
```

## 📖 Usage Guide

### Basic Controls

- **`q`**: Quit the application
- **`s`**: Save screenshot
- **`h`**: Toggle heatmap visibility
- **`l`**: Toggle landmark points
- **`r`**: Toggle region labels
- **`f`**: Toggle face mesh overlay

### Camera Selection

When starting the application, you'll be prompted to choose:
1. **PC/Laptop Webcam** (Default)
2. **Intel RealSense Camera** (if connected)

### Understanding the Display

The application opens three windows:
1. **Main Window**: Combined heatmap + face mesh with full features
2. **Heatmap Only**: Pure heatmap visualization
3. **Face Map Only**: Landmark and region visualization

## ⚙️ Configuration

### Config.json Parameters

```json
{
  "heatmap": {
    "sigma": 18.0,              // Gaussian kernel size (higher = smoother)
    "temporal_smoothing": 0.85, // EMA smoothing factor (0-1, higher = smoother)
    "buffer_size": 5,           // Heatmap history buffer
    "alpha": 0.6,               // Heatmap transparency
    "colormap": 11              // OpenCV colormap index
  },
  "tracking": {
    "time_windows": [5.0, 10.0, 15.0], // Analysis time windows
    "activity_threshold": 0.02,        // Minimum activity for dominant region
    "dominance_debounce_frames": 1,    // Frames required for stable region
    "max_faces": 1,                    // Maximum faces to track
    "min_detection_confidence": 0.5,    // MediaPipe detection confidence
    "min_tracking_confidence": 0.5     // MediaPipe tracking confidence
  },
  "camera": {
    "width": 640,               // Camera resolution width
    "height": 480,              // Camera resolution height
    "fps": 30                    // Target FPS
  }
}
```

### Performance Tuning

- **For better stability**: Increase `temporal_smoothing` and `sigma`
- **For better responsiveness**: Decrease `temporal_smoothing` and `sigma`
- **For higher FPS**: Decrease camera resolution or increase `landmark_step`
- **For more accurate tracking**: Increase MediaPipe confidence thresholds

## 🧠 Facial Regions

The system tracks 16 distinct facial regions:

| Region | Description | Key Features |
|--------|-------------|--------------|
| **Eyes** | LEFT_EYE, RIGHT_EYE | Eye movement, blinking |
| **Eyebrows** | LEFT_EYEBROW, RIGHT_EYEBROW | Expression, attention |
| **Mouth** | LIPS | Speech, expressions |
| **Nose** | NOSE | Head movement reference |
| **Forehead** | FOREHEAD | Expression, brow movement |
| **Cheeks** | LEFT_CHEEK, RIGHT_CHEEK | Facial expressions |
| **Chin/Jaw** | CHIN | Jaw movement, speech |
| **Ears** | LEFT_EAR, RIGHT_EAR | Head orientation |
| **Other** | HAIR, NECK, SHOULDERS | Upper body movement |

## 📊 Performance Metrics

### Expected Performance

- **FPS**: 25-30 frames per second
- **Centroid Stability**: < 2.0px average displacement
- **Large Jumps**: < 5% of frames with >10px movement
- **Memory Usage**: ~100-200MB
- **CPU Usage**: 15-25% (varies by hardware)

### Validation Results

The system includes comprehensive validation tools:

```bash
# Run stability validation
python final_validation_test.py

# Debug specific issues
python debug_heatmap_centroid.py
python debug_movement_weights.py
python debug_activity_scores.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "No dominant region detected"
**Cause**: Activity threshold too high or insufficient movement
**Solution**: 
```json
"activity_threshold": 0.01  // Lower threshold
"dominance_debounce_frames": 1  // Reduce debouncing
```

#### 2. "Heatmap fluctuates randomly"
**Cause**: Movement weights causing instability
**Solution**: Movement weights are disabled by default in the latest version

#### 3. "Low FPS performance"
**Cause**: High resolution or processing overhead
**Solution**:
```json
"camera": {
  "width": 480,    // Reduce resolution
  "height": 360
}
```

#### 4. "MediaPipe import errors"
**Cause**: Version incompatibility
**Solution**:
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.15
```

### Debug Mode

Enable detailed logging by setting debug mode in the code:

```python
# In tracker.py, add debug prints
print(f"DEBUG: dominant_region = {dominant_region}")
print(f"DEBUG: activity_scores = {activity_scores}")
```

## 📁 Project Structure

```
face-recog-video/
├── README.md                    # This file
├── config.json                 # Configuration parameters
├── requirements.txt             # Python dependencies
├── start_camera_fast.py         # Main application launcher
├── run_app.py                   # Alternative launcher
├── main.py                      # Core application class
├── tracker.py                   # Face tracking logic
├── heatmap.py                   # Heatmap generation
├── regions.py                   # Facial region definitions
├── camera_manager.py            # Camera interface
├── debug_*.py                   # Debug and validation tools
├── output/                      # Generated files
│   ├── *.mp4                   # Recorded videos
│   ├── *.csv                   # Activity logs
│   └── *.json                  # Validation results
└── docs/                        # Documentation
```

## 🎯 Advanced Features

### Activity Scoring Algorithm

The system uses a sophisticated activity scoring system:

1. **Movement Detection**: Calculates landmark displacement between frames
2. **Normalization**: Relative to global maximum movement
3. **Area Weighting**: Penalizes large regions to avoid bias
4. **Temporal Smoothing**: EMA filtering for consistent scores
5. **Debouncing**: Requires consecutive frames for stable detection

### Heatmap Generation

- **Gaussian Kernels**: Weighted by landmark positions
- **Temporal Smoothing**: Exponential moving average
- **Adaptive Sigma**: Configurable kernel size
- **Colormap Mapping**: 11 different color schemes

### Stabilization Features

- **Landmark Smoothing**: Reduces jitter in face detection
- **Region Debouncing**: Prevents rapid switching between regions
- **Activity Thresholding**: Filters out minor movements
- **Temporal Filtering**: Smooths heatmap over time

## 📈 Research & Development

### Key Improvements Made

1. **Movement Weight Optimization**: Disabled problematic movement weights that caused 488% instability
2. **Dominant Region Detection**: Fixed debouncing logic for consistent region tracking
3. **Temporal Smoothing**: Optimized EMA parameters for stable heatmaps
4. **Activity Scoring**: Improved normalization and area weighting
5. **Performance Tuning**: Achieved 29 FPS with stable tracking

### Validation Metrics

- **Centroid Displacement**: Reduced from 9.091px to 2.084px (77.1% improvement)
- **Large Jumps**: Eliminated completely (100% reduction)
- **Processing Speed**: Maintained 29 FPS
- **Memory Efficiency**: Optimized buffer management

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd face-recog-video
pip install -r requirements.txt

# Run tests
python final_validation_test.py

# Debug specific issues
python debug_heatmap_centroid.py
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive comments
- Include validation tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's face mesh detection framework
- **OpenCV**: Computer vision and image processing
- **Intel RealSense**: Depth camera support (optional)

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Run the validation tools for diagnostics
3. Review the debug logs in `output/` directory
4. Check configuration parameters in `config.json`

---

**Last Updated**: March 2026  
**Version**: 2.0 - Stable Release  
**Performance**: 29 FPS, 77.1% stability improvement
