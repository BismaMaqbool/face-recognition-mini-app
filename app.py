import argparse
import cv2
import sys
import os


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


def main():
    args = parse_args()
    source = get_video_source(args.source)

    # Validate db folder exists
    if not os.path.isdir(args.db):
        print(f"Error: Database folder not found: {args.db}")
        sys.exit(1)

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        sys.exit(1)

    print(f"Video source: {args.source}")
    print(f"Database folder: {args.db}")
    print("Press 'q' to quit.")

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

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
