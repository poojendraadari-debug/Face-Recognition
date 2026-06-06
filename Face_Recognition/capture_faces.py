import os
import cv2
from utils import ensure_directories, extract_face


def collect_face_images():
    ensure_directories()

    name = input("Enter the person's full name: ").strip()
    if not name:
        print("Name is required.")
        return

    gender = input("Enter gender (male/female): ").strip().lower()
    if gender not in {"male", "female"}:
        print("Gender must be 'male' or 'female'.")
        return

    folder_name = f"{name}_{gender}"
    target_folder = os.path.join("dataset", folder_name)
    os.makedirs(target_folder, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Unable to open webcam. Make sure the camera is connected and not used by another application.")
        return

    cv2.namedWindow("Capture Faces", cv2.WINDOW_NORMAL)
    print("Webcam started. Click the Capture Faces window once, then press 'c' or Space to save a face image. Press 'q' or Esc to quit.")

    count = len([n for n in os.listdir(target_folder) if n.lower().endswith(('.png', '.jpg', '.jpeg'))])
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from webcam.")
            break

        face, coords = extract_face(frame)
        if face is not None:
            x, y, w, h = coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press C or Space to capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No face found. Move closer and center your face.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Captured: {count}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1)
        if key != -1:
            key = key & 0xFF
        if key == ord("q") or key == 27:
            break

        if key == ord("c") or key == ord("C") or key == 32:
            if face is not None:
                file_path = os.path.join(target_folder, f"face_{count + 1:03d}.png")
                cv2.imwrite(file_path, face)
                count += 1
                print(f"Saved {file_path}")
            else:
                print("No face found. Please move closer to the camera and try again.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images for {name}.")


if __name__ == "__main__":
    collect_face_images()
