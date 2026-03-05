#!/usr/bin/env python3
"""
Face Tracking App Launcher
Cross-platform startup script for beginners
Works on Windows and Linux
"""

import sys
import os
import subprocess
import platform
import stat
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def fix_file_permissions():
    """Fix file permissions for cross-platform compatibility"""
    print("🔐 Checking and fixing file permissions...")
    
    # Files that need execute permissions on Linux/Mac
    executable_files = [
        "run_app.py",
        "start_linux.sh",
        "start_camera_fast.py", 
        "start_camera.py"
    ]
    
    # Fix permissions for Python scripts and shell scripts
    for filename in executable_files:
        file_path = Path(filename)
        if file_path.exists():
            try:
                # Get current permissions
                current_mode = file_path.stat().st_mode
                
                # Add execute permission for user on Linux/Mac (755 style)
                if platform.system() != "Windows":
                    # Add user execute permission if not present
                    if not (current_mode & stat.S_IXUSR):
                        os.chmod(file_path, current_mode | stat.S_IXUSR)
                        print(f"✅ Made {filename} executable")
                    else:
                        print(f"✅ {filename} already executable")
                else:
                    print(f"✅ {filename} permissions OK (Windows)")
                    
            except Exception as e:
                print(f"⚠️  Could not set permissions for {filename}: {e}")
    
    # Ensure output directory is writable
    output_dir = Path("output")
    try:
        output_dir.mkdir(exist_ok=True)
        if platform.system() != "Windows":
            # Ensure full read/write/execute for user, read/execute for group/others
            os.chmod(output_dir, 0o755)
        print("✅ Output directory permissions set")
    except Exception as e:
        print(f"⚠️  Could not set output directory permissions: {e}")
    
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Checking and installing dependencies...")
    
    packages = [
        "opencv-python==4.8.1.78",
        "mediapipe==0.10.9", 
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scipy==1.11.1"
    ]
    
    for package in packages:
        try:
            __import__(package.split('==')[0].replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📥 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def check_camera():
    """Check if camera is available"""
    print("📷 Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working")
                return True
            else:
                print("⚠️  Camera found but couldn't capture image")
                return False
        else:
            print("❌ No camera found")
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def start_app():
    """Start the face tracking application"""
    print("\n🚀 Starting Face Tracking App...")
    print("=" * 50)
    
    # Try the fast version first (optimized)
    scripts_to_try = ["start_camera_fast.py", "start_camera.py"]
    
    for script in scripts_to_try:
        script_path = Path(script)
        if script_path.exists():
            # Ensure script has execute permissions on Linux/Mac
            if platform.system() != "Windows":
                try:
                    current_mode = script_path.stat().st_mode
                    if not (current_mode & stat.S_IXUSR):
                        os.chmod(script_path, current_mode | stat.S_IXUSR)
                        print(f"🔐 Made {script} executable")
                except Exception as e:
                    print(f"⚠️  Could not set execute permission: {e}")
            
            print(f"🎯 Starting {script}...")
            try:
                if platform.system() == "Windows":
                    subprocess.run([sys.executable, str(script_path)], check=True)
                else:
                    subprocess.run([sys.executable, str(script_path)], check=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to start {script}: {e}")
                continue
            except KeyboardInterrupt:
                print("\n🛑 App stopped by user")
                return True
    
    print("❌ No startup script found")
    return False

def show_instructions():
    """Show usage instructions"""
    print("\n📖 Face Tracking App Instructions:")
    print("=" * 40)
    print("🎮 Controls:")
    print("   'q' - Quit the application")
    print("   's' - Save screenshot")
    print("   'h' - Toggle heatmap display")
    print("   'l' - Toggle landmarks display")
    print("   'r' - Toggle regions display")
    print("   'f' - Toggle fullscreen")
    print("\n📁 Output files will be saved in 'output/' folder:")
    print("   • Video recording (MP4)")
    print("   • CSV data log")
    print("\n💡 Tips:")
    print("   • Sit in good lighting")
    print("   • Face the camera directly")
    print("   • Stay within camera view")
    print("   • Press 'q' to stop recording")

def main():
    """Main launcher function"""
    print("🎭 Face Tracking App Launcher")
    print("=" * 40)
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📂 Working directory: {script_dir}")
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("File Permissions", fix_file_permissions),
        ("Dependencies", install_dependencies),
        ("Camera", check_camera)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
            print(f"❌ {check_name} check failed")
            if check_name == "Python Version":
                print("Please install Python 3.7+ from https://python.org")
            elif check_name == "File Permissions":
                print("Try running: chmod +x *.py *.sh (Linux/Mac)")
                print("On Windows, check if files are not blocked by antivirus")
            elif check_name == "Dependencies":
                print("Try running: pip install -r requirements.txt")
            elif check_name == "Camera":
                print("Please connect a camera and check permissions")
    
    if not all_passed:
        print("\n❌ Some checks failed. Please fix the issues above.")
        input("Press Enter to exit...")
        return
    
    # Show instructions
    show_instructions()
    
    # Ask user to continue
    try:
        response = input("\n🚀 Ready to start? (Press Enter to continue, or 'q' to quit): ")
        if response.lower() == 'q':
            print("👋 Goodbye!")
            return
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Start the app
    try:
        success = start_app()
        if success:
            print("\n✅ App completed successfully!")
        else:
            print("\n❌ App failed to start")
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n📁 Check the 'output/' folder for saved files!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
