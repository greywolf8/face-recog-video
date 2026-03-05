# Face Tracking App - Quick Start Guide

## 🚀 Easy Launch (For Beginners)

### Windows Users:
1. **Double-click** `start_windows.bat`
2. Follow the on-screen instructions
3. Enjoy the app!

### Linux Users:
1. **Double-click** `start_linux.sh` (or run `./start_linux.sh` in terminal)
2. Follow the on-screen instructions
3. Enjoy the app!

### Any Platform:
1. Run `python run_app.py` in terminal
2. The launcher will check everything automatically

## 📋 What the Launcher Does:

✅ **Checks Python version** (needs 3.7+)  
✅ **Installs dependencies** automatically  
✅ **Tests camera** availability  
✅ **Shows instructions** for controls  
✅ **Starts the app** with optimal settings  

## 🎮 App Controls:

- **'q'** - Quit application  
- **'s'** - Save screenshot  
- **'h'** - Toggle heatmap  
- **'l'** - Toggle landmarks  
- **'r'** - Toggle regions  
- **'f'** - Toggle fullscreen  

## 📁 Output Files:

All files are saved in the `output/` folder:
- 📹 Video recording (MP4)
- 📊 CSV data log
- 📸 Screenshots (when you press 's')

## 💡 Tips for Best Results:

- Sit in good, even lighting
- Face the camera directly  
- Stay within the camera view
- Keep your face visible (no masks/hats)
- Press 'q' when you're done to save files

## 🔧 If Something Goes Wrong:

**Python not found?**
- Windows: Install from https://python.org
- Linux: `sudo apt install python3 python3-pip`

**Camera not working?**
- Check camera is connected
- Allow camera permissions
- Close other apps using the camera

**Dependencies failed?**
- Run: `pip install -r requirements.txt`
- Or let the launcher fix it automatically

## 📞 Need Help?

The launcher provides detailed error messages and suggestions.
If you're still stuck, check the output folder for error logs.

---

**That's it! Just double-click and go!** 🎉
