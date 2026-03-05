#!/usr/bin/env python3
"""
Quick Start Camera Script
Automatically turns on camera and starts face heatmap tracking with video recording
"""

import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import FaceHeatmapApp

def main():
    """Start camera and begin recording immediately"""
    print("🎥 Starting Face Heatmap Tracking System...")
    print("📹 Camera will turn on automatically")
    print("💾 Video will be saved automatically")
    print("⏹️  Press 'q' to stop recording and quit")
    print("-" * 50)
    
    # Create app with auto-start settings
    app = FaceHeatmapApp(
        camera_index=0,        # Default camera
        save_video=True,       # Auto-save video
        output_dir="output",   # Save to output folder
        show_landmarks=True,   # Show face landmarks
        show_heatmap=True,     # Show heatmap overlay
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
        print("✅ Camera stopped and files saved")

if __name__ == "__main__":
    main()
