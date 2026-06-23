![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-pose-green)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

# Biomechanics Motion and Analysis System for Boxing

Analyzes boxing technique in real-time using a single laptop camera — no expensive hardware needed. Powered by MediaPipe pose estimation, it scores your punches and gives actionable feedback compared to professional reference poses.

## Features
- Real-time pose estimation (MediaPipe)
- Video upload and offline analysis
- Frame-by-Frame Analysis and scoring (jab, cross, hook, uppercut)
- Visual feedback and interactive charts (Plotly)
- Configurable processing rate and frame limits for performance

## Requirements
- Python 3.8+
- See `requirements.txt` for the tested dependency set

## Installation
1. Create and activate a virtual environment (recommended):

```bash
# Linux/Mac:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate    
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
- Run the Streamlit app (development):

```bash
streamlit run app.py --server.address localhost --server.port 5000
```

- For quick script-based processing (examples):

```bash
python video_processor.py --input path/to/video.mp4 --output results.json
```

Adjust settings such as frame rate and max frames inside `video_processor.py` or via the Streamlit UI.

## Project Structure
- `app.py` — Streamlit web app and UI entrypoint
- `video_processor.py` — Video I/O, frame extraction, and batch analysis
- `pose_estimator.py` — MediaPipe pose detection wrapper and utilities
- `boxing_analyzer.py` — Technique analysis logic, scoring, and punch detection
- `reference_poses.py` — Professional reference pose templates and tolerances
- `utils.py` — Shared helpers (IO, angle math, plotting helpers)
- `requirements.txt` — Python dependencies

## Configuration
- Tweak frame-rate, model complexity, and max frames in `video_processor.py` and `pose_estimator.py` for a speed/accuracy tradeoff.

## Troubleshooting
- If pose detection is poor, increase frame resolution or reduce processing FPS.
- For high memory use, lower `MAX_FRAMES` or process shorter clips.

## Contributing
- Open an issue or submit a pull request with a clear description and tests or reproducible steps. Thank you :)