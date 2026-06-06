import os
import json
import cv2
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def ensure_directories():
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def extract_face(frame, size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = FACE_CASCADE.detectMultiScale(
            small_gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20),
        )
        if len(faces) > 0:
            faces = [(x * 2, y * 2, w * 2, h * 2) for (x, y, w, h) in faces]

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, size)
    return face, (x, y, w, h)


def prepare_face(face):
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    return face


def load_dataset(dataset_dir="dataset"):
    name_to_label = {}
    label_to_name = {}
    gender_to_label = {"male": 0, "female": 1}

    X, y_name, y_gender = [], [], []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for folder_name in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path) or "_" not in folder_name:
            continue

        person_name, gender_value = folder_name.rsplit("_", 1)
        gender_value = gender_value.lower()
        if gender_value not in gender_to_label:
            continue

        if person_name not in name_to_label:
            label = len(name_to_label)
            name_to_label[person_name] = label
            label_to_name[label] = person_name

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (64, 64))
            X.append(image)
            y_name.append(name_to_label[person_name])
            y_gender.append(gender_to_label[gender_value])

    if len(X) == 0:
        raise ValueError("No training images found in the dataset. Add faces first.")

    X = np.array(X, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y_name = np.array(y_name, dtype="int32")
    y_gender = np.array(y_gender, dtype="float32")

    return X, y_name, y_gender, name_to_label, label_to_name


def save_label_map(label_map, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
