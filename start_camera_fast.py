#!/usr/bin/env python3
"""
Performance Optimized Camera Script
Optimized for higher FPS with reduced computational overhead
"""

import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import FaceHeatmapApp
from camera_manager import CameraManager

def main():
    """Start camera with performance optimizations"""
    print("🚀 Starting OPTIMIZED Face Heatmap Tracking...")
    print("⚡ Performance mode enabled for higher FPS")
    print("📹 Camera: 640x480 (optimized)")
    print("💾 Video will be saved automatically")
    print("⏹️  Press 'q' to stop recording and quit")
    print("-" * 50)
    
    # Initialize camera manager and let user select camera
    camera_manager = CameraManager()
    if not camera_manager.initialize():
        print("❌ Failed to initialize any camera")
        return
    
    camera_type = camera_manager.get_camera_type()
    print(f"📷 Using: {camera_type}")
    
    # Create app with performance optimizations
    app = FaceHeatmapApp(
        camera_manager=camera_manager,  # Pass camera manager instead of index
        save_video=True,       # Auto-save video
        output_dir="output",   # Save to output folder
        show_landmarks=True,   # Show face landmarks
        show_heatmap=True,     # Show heatmap (optimized)
        show_regions=True      # Show region labels
    )
    
    # Run the application
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Ensure camera is released
        camera_manager.release()
        print("✅ Camera stopped and files saved")

if __name__ == "__main__":
    main()
