import argparse
import cv2
import sys
import os
import numpy as np
from deepface import DeepFace


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection + Verification Mini App")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--db", default="./known_faces",
                        help="Path to known faces database folder")
    return parser.parse_args()


def get_video_source(source_str):
    """Convert source argument to int (webcam) or keep as string (file path)."""
    try:
        return int(source_str)
    except ValueError:
        if not os.path.isfile(source_str):
            print(f"Error: Video file not found: {source_str}")
            sys.exit(1)
        return source_str


def load_known_faces(db_path):
    """
    Load face embeddings from the known_faces database folder.

    Args:
        db_path: Path to the database folder (e.g., ./known_faces)

    Returns:
        List of dictionaries: [{"name": "PersonName", "embedding": [...]}, ...]
    """
    known_faces = []

    if not os.path.isdir(db_path):
        print(f"Warning: Database folder not found: {db_path}")
        return known_faces

    print("\nLoading known faces...")

    # Iterate through person folders
    for person_name in os.listdir(db_path):
        person_path = os.path.join(db_path, person_name)

        if not os.path.isdir(person_path):
            continue

        # Iterate through images for this person
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)

            # Skip non-image files
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                # Extract embedding using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                # DeepFace.represent returns a list; take the first face
                if embedding_objs:
                    embedding = embedding_objs[0]["embedding"]
                    known_faces.append({
                        "name": person_name,
                        "embedding": np.array(embedding)
                    })
                    print(f"  ✓ Loaded: {person_name}/{img_file}")

            except Exception as e:
                print(f"  ✗ Failed to load {person_name}/{img_file}: {e}")

    print(f"\nTotal faces loaded: {len(known_faces)}")
    return known_faces


def cosine_distance(embedding1, embedding2):
    """
    Compute cosine distance between two embeddings.
    Cosine distance = 1 - cosine_similarity

    Range: [0, 2] where 0 = identical, 2 = opposite
    """
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    cosine_sim = dot_product / (norm1 * norm2)

    # Return cosine distance
    return 1 - cosine_sim


def find_best_match(face_embedding, known_faces, threshold=0.5):
    """
    Find the best matching known face using cosine distance.

    Args:
        face_embedding: Embedding of the detected face
        known_faces: List of known face dictionaries
        threshold: Maximum distance to consider a match (default 0.5)

    Returns:
        Name of the matched person or "Unknown"
    """
    if not known_faces:
        return "Unknown"

    min_distance = float('inf')
    best_match = "Unknown"

    for known_face in known_faces:
        distance = cosine_distance(face_embedding, known_face["embedding"])

        if distance < min_distance:
            min_distance = distance
            best_match = known_face["name"]

    # Return name if below threshold, otherwise "Unknown"
    return best_match if min_distance < threshold else "Unknown"


def main():
    args = parse_args()
    source = get_video_source(args.source)

    # Validate db folder exists
    if not os.path.isdir(args.db):
        print(f"Error: Database folder not found: {args.db}")
        sys.exit(1)

    # PHASE 4: Load known faces embeddings
    known_faces = load_known_faces(args.db)

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        sys.exit(1)

    print(f"\nVideo source: {args.source}")
    print(f"Database folder: {args.db}")
    print("Press 'q' to quit.\n")

    # PHASE 5: Performance optimization - process recognition every N frames
    PROCESS_EVERY_N_FRAMES = 5
    frame_count = 0
    face_labels = {}  # Store labels for each face: {face_index: "Name"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # PHASE 5: Perform recognition every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            face_labels.clear()

            for idx, (x, y, w, h) in enumerate(faces):
                # Crop the face region
                face_img = frame[y:y+h, x:x+w]

                try:
                    # Extract embedding using DeepFace
                    embedding_objs = DeepFace.represent(
                        img_path=face_img,
                        model_name="Facenet",
                        enforce_detection=False
                    )

                    if embedding_objs:
                        face_embedding = embedding_objs[0]["embedding"]
                        # Find best match
                        name = find_best_match(face_embedding, known_faces, threshold=0.5)
                        face_labels[idx] = name
                    else:
                        face_labels[idx] = "Unknown"

                except Exception as e:
                    face_labels[idx] = "Unknown"

        # Draw bounding boxes and labels
        for idx, (x, y, w, h) in enumerate(faces):
            # Get label (use cached label if available)
            label = face_labels.get(idx, "Unknown")

            # Choose color based on match
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
