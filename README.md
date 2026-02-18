# Face Detection + Verification Mini App

A Python terminal application that detects faces in real-time from a webcam or video file using OpenCV, with a planned face verification pipeline using DeepFace.

## Current Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Setup & Dependencies | Done |
| 2 | CLI Arguments + Video Capture | Done |
| 3 | Face Detection with OpenCV | Done |
| 4 | Load Known Faces & Embeddings | Done |
| 5 | Face Recognition & Labeling | Done |

### Phase 1 — Project Setup & Dependencies
Initialized the project structure, defined dependencies in `requirements.txt`, and created the `known_faces/` database folder.

### Phase 2 — CLI Arguments + Video Capture
Added `argparse` support for `--source` (webcam or video path) and `--db` (known faces folder). The app opens the video source and displays frames in a live window.

### Phase 3 — Face Detection with OpenCV
Integrated Haar cascade face detection. The app draws green bounding boxes around detected faces in real-time.

### Phase 4 — Load Known Faces & Embeddings
Implemented face database loading on startup. The app scans the `known_faces/` directory for person subfolders, extracts face embeddings using **DeepFace** with the **Facenet** model, and stores them in memory. Each person can have multiple images for more robust recognition.

**Key Features:**
- Pre-computes face embeddings once at startup (no repeated extraction during runtime)
- Supports multiple images per person
- Handles errors gracefully (skips corrupt or invalid images)
- Prints loading progress and summary

### Phase 5 — Face Recognition & Labeling
Implemented real-time face recognition with identity matching and labeling. Detected faces are compared against known embeddings using **cosine distance** with a threshold of **0.5**.

**Key Features:**
- Extracts embeddings for detected faces and matches them against the known faces database
- Uses cosine distance for robust comparison (scale-invariant, ideal for high-dimensional embeddings)
- Labels recognized faces with their names in **green boxes**
- Labels unrecognized faces as "Unknown" in **red boxes**
- **Performance optimization:** Processes recognition every **5 frames** to maintain smooth real-time performance
- Label caching between frames for consistent display

**Why Cosine Distance?**
- Scale-invariant (doesn't depend on vector magnitude)
- More robust for high-dimensional embeddings (Facenet produces 128-D vectors)
- Standard metric in face recognition tasks

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
- `*.jpg`, `*.jpeg`, `*.png`, `*.gif`, `*.webp`, `*.svg`, etc. — Image files (prevents accidentally pushing personal photos from `known_faces/`)
- `.DS_Store` — macOS system files

Make sure your virtual environment is created **inside** the project folder (e.g., `.venv/`) so it gets ignored automatically.

**Note:** Since image files are now ignored, the `known_faces/` folder structure will be committed, but the actual face images will not be pushed to GitHub. This protects privacy and keeps the repository size small.
