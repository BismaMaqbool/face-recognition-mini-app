import argparse
import cv2
import sys
import os
import time
import numpy as np
from deepface import DeepFace


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESS_EVERY_N_FRAMES = 5
MATCH_THRESHOLD = 0.5
MIN_FACE_SIZE = (30, 30)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for video source and known-faces database."""
    parser = argparse.ArgumentParser(
        description="Face Detection + Verification Mini App"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0 for webcam, or path to video file",
    )
    parser.add_argument(
        "--db", default="./known_faces",
        help="Path to known faces database folder",
    )
    return parser.parse_args()


def get_video_source(source_str):
    """Convert source argument to int (webcam index) or validate file path."""
    try:
        return int(source_str)
    except ValueError:
        if not os.path.isfile(source_str):
            print(f"Error: Video file not found: {source_str}")
            sys.exit(1)
        return source_str


# ---------------------------------------------------------------------------
# Database loading  (Phase 4)
# ---------------------------------------------------------------------------
def load_known_faces(db_path):
    """
    Scan the known-faces folder and build an in-memory embedding database.

    Expected structure:
        db_path/
            PersonName/
                image1.jpg
                image2.png

    Returns:
        List of dicts: [{"name": str, "embedding": np.ndarray}, ...]
    """
    known_faces = []

    if not os.path.isdir(db_path):
        print(f"Warning: Database folder not found: {db_path}")
        return known_faces

    print("\nLoading known faces...")

    person_count = 0

    for person_name in sorted(os.listdir(db_path)):
        person_path = os.path.join(db_path, person_name)

        if not os.path.isdir(person_path):
            continue

        person_count += 1

        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(person_path, img_file)

            try:
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False,
                )

                if embedding_objs:
                    known_faces.append({
                        "name": person_name,
                        "embedding": np.array(embedding_objs[0]["embedding"]),
                    })
                    print(f"  Loaded: {person_name}/{img_file}")

            except Exception as e:
                print(f"  Failed: {person_name}/{img_file}: {e}")

    print(f"\nLoaded {len(known_faces)} embedding(s) for {person_count} person(s).")

    if not known_faces:
        print("Warning: No embeddings loaded. All faces will be labeled 'Unknown'.")

    return known_faces


# ---------------------------------------------------------------------------
# Embedding comparison  (Phase 5 helpers)
# ---------------------------------------------------------------------------
def cosine_distance(embedding1, embedding2):
    """
    Compute cosine distance between two vectors.

    Returns a value in [0, 2] where 0 = identical, 2 = opposite.
    Returns 1.0 (no match) when either vector has zero magnitude.
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 1.0

    cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return 1 - cosine_sim


def find_best_match(face_embedding, known_faces, threshold=MATCH_THRESHOLD):
    """
    Compare a face embedding against all known faces.

    Returns the name of the closest match if the cosine distance is below
    the threshold, otherwise returns 'Unknown'.
    """
    if not known_faces:
        return "Unknown"

    face_embedding = np.asarray(face_embedding)

    min_distance = float("inf")
    best_match = "Unknown"

    for known_face in known_faces:
        distance = cosine_distance(face_embedding, known_face["embedding"])

        if distance < min_distance:
            min_distance = distance
            best_match = known_face["name"]

    return best_match if min_distance < threshold else "Unknown"


# ---------------------------------------------------------------------------
# Detection & recognition  (Phase 3 / 5 helpers)
# ---------------------------------------------------------------------------
def detect_faces(frame, face_cascade):
    """
    Run Haar cascade face detection on a single frame.

    Returns:
        Tuple array of (x, y, w, h) for each detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )
    return faces


def recognize_face(face_img, known_faces):
    """
    Extract an embedding from a cropped face image and find the best match.

    Returns the matched person's name or 'Unknown'.
    """
    try:
        embedding_objs = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False,
        )

        if embedding_objs:
            return find_best_match(embedding_objs[0]["embedding"], known_faces)
    except Exception:
        pass

    return "Unknown"


# ---------------------------------------------------------------------------
# Drawing helpers  (Phase 6)
# ---------------------------------------------------------------------------
def draw_face_label(frame, x, y, w, h, label):
    """Draw a colored bounding box and name label on the frame."""
    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)
    cv2.putText(
        frame, label, (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def draw_fps(frame, fps):
    """Display an FPS counter in the top-left corner of the frame."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    source = get_video_source(args.source)

    # Validate database folder
    if not os.path.isdir(args.db):
        print(f"Error: Database folder not found: {args.db}")
        sys.exit(1)

    # Phase 4 — load known-face embeddings
    known_faces = load_known_faces(args.db)

    # Load Haar cascade
    cascade_path = os.path.join(
        os.path.expanduser("~"), ".deepface", "weights",
        "haarcascade_frontalface_default.xml",
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Failed to load Haar cascade classifier.")
        sys.exit(1)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        sys.exit(1)

    print(f"\nVideo source: {args.source}")
    print(f"Database folder: {args.db}")
    print("Press 'q' to quit.\n")

    # Frame-processing state
    frame_count = 0
    face_labels = {}
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces every frame (lightweight)
        faces = detect_faces(frame, face_cascade)

        # Run recognition only every N frames (expensive)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_labels.clear()
            for idx, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y + h, x:x + w]
                face_labels[idx] = recognize_face(face_img, known_faces)

        # Draw results
        for idx, (x, y, w, h) in enumerate(faces):
            label = face_labels.get(idx, "Unknown")
            draw_face_label(frame, x, y, w, h, label)

        # FPS counter
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time
        draw_fps(frame, fps)

        cv2.imshow("Face Recognition", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
