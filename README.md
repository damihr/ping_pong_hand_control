# 🏓 Air Pong - Hand Control

A hand-controlled Pong game that works both as a Python desktop application and as a web application on GitHub Pages!

## 🌐 Web Version (GitHub Pages)

**Play online:** https://damihr.github.io/ping_pong_hand_control/

The web version uses MediaPipe for JavaScript and runs entirely in your browser - no installation needed!

### Features:
- **1 Player Mode** - Play against an AI bot
- **2 Player Mode** - Play with a friend using both hands
- Real-time hand tracking via webcam
- Smooth paddle movement with hand gestures
- Beautiful UI with webcam background

### Controls:
- Move your **left hand up/down** to control the left paddle
- Move your **right hand up/down** to control the right paddle (2 player mode)
- Press **M** to switch between 1 player and 2 player modes
- Press **Q** to return to menu

## 💻 Python Desktop Version

For running locally with Python.

### Requirements

- **Python 3.9** (recommended for macOS stability with MediaPipe)
- Webcam/camera
- macOS/Linux/Windows

### Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:damihr/ping_pong_hand_control.git
   cd ping_pong_hand_control
   ```

2. **Create a virtual environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Usage

1. **Run the Python version:**
   ```bash
   python game.py
   ```

2. **Or use the web version:**
   - Open `index.html` in your browser
   - Or deploy to GitHub Pages (see below)

## 🚀 Deploying to GitHub Pages

1. **Push the repository to GitHub**
2. **Go to repository Settings → Pages**
3. **Select source:** Deploy from a branch
4. **Select branch:** `main` and folder: `/ (root)`
5. **Save**

Your game will be live at: `https://damihr.github.io/ping_pong_hand_control/`

## 📁 Files

- **`game.py`** - Python desktop version (OpenCV + MediaPipe)
- **`index.html`** - Web version (MediaPipe JS + Canvas)
- **`requirements.txt`** - Python dependencies

## 🎮 Game Modes

### 1 Player Mode
- Control left paddle with your left hand
- AI bot controls right paddle
- Bot difficulty: Perfect (predicts ball trajectory)

### 2 Player Mode
- Left hand controls left paddle
- Right hand controls right paddle
- Perfect for playing with a friend!

## 🛠️ Troubleshooting

### Web Version
- **Camera not working:** Make sure you allow camera access when prompted
- **Hands not detected:** Ensure good lighting and keep hands visible
- **Performance issues:** Close other browser tabs, use Chrome/Edge for best performance

### Python Version
- **MediaPipe errors:** Use Python 3.9 and ensure dependencies are installed
- **Camera not found:** Check camera permissions and try different camera index
- **Fullscreen issues:** Press 'Q' to quit, or Alt+F4 on Windows

## 📝 Notes

- The web version uses MediaPipe JavaScript SDK (runs entirely client-side)
- The Python version uses MediaPipe Python SDK (requires installation)
- Both versions support the same game modes and features
- Web version is optimized for modern browsers (Chrome, Edge, Firefox, Safari)

## License

MIT License

