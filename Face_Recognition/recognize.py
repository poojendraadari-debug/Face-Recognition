import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_face, prepare_face


MODEL_PATH = os.path.join("models", "face_recognition_rnn.h5")
LABEL_PATH = os.path.join("models", "label_map.json")


def load_label_map():
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload.get("label_to_name", {}).items()}


def recognize():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print("Model or label map not found. Run train_model.py first.")
        return

    model = load_model(MODEL_PATH)
    label_map = load_label_map()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Unable to open webcam. Make sure the camera is connected and not used by another application.")
        return

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    print("Recognition started. Press 'q' or Esc to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from webcam.")
            break

        face, coords = extract_face(frame)
        if face is not None:
            x, y, w, h = coords
            input_face = prepare_face(face)
            input_face = np.expand_dims(input_face, axis=0)

            identity_pred, gender_pred = model.predict(input_face, verbose=0)
            identity_index = int(np.argmax(identity_pred[0]))
            gender_value = float(gender_pred[0][0])
            name = label_map.get(identity_index, "Unknown")
            gender_text = "Female" if gender_value >= 0.5 else "Male"

            label = f"{name} | {gender_text}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "Press q/Esc to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No face detected. Center your face in the webcam.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(30)
        if key != -1:
            key = key & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
