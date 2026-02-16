# Face Detection + Verification Mini App

A Python terminal application that detects faces in real-time from a webcam or video file using OpenCV, with a planned face verification pipeline using DeepFace.

## Current Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup & Dependencies | Done |
| 2 | CLI Arguments + Video Capture | Done |
| 3 | Face Detection with OpenCV | Done |

### Phase 1 — Project Setup & Dependencies
Initialized the project structure, defined dependencies in `requirements.txt`, and created the `known_faces/` database folder.

### Phase 2 — CLI Arguments + Video Capture
Added `argparse` support for `--source` (webcam or video path) and `--db` (known faces folder). The app opens the video source and displays frames in a live window.

### Phase 3 — Face Detection with OpenCV
Integrated Haar cascade face detection. The app draws green bounding boxes around detected faces in real-time.

## Project Structure

```
Face_Recog/
├── .gitignore
├── requirements.txt
├── app.py
├── README.md
└── known_faces/
    └── Leen/
        └── (add face images here)
```

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run with webcam:**
```bash
python app.py
```

**Run with a video file:**
```bash
python app.py --source path/to/video.mp4
```

**Specify a custom known faces folder:**
```bash
python app.py --source 0 --db ./known_faces
```

Press `q` to quit the application.

## Git & GitHub Notes

The `.gitignore` file is configured to exclude the following from version control:

- `__pycache__/` and `*.pyc` — Python cache files
- `.venv/`, `venv/`, `env/` — Virtual environment folders
- `.env` — Environment variable files
- `*.mp4`, `*.avi` — Video files (large binaries)
- `.DS_Store` — macOS system files

Make sure your virtual environment is created **inside** the project folder (e.g., `.venv/`) so it gets ignored automatically.
