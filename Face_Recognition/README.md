# Face Recognition with Webcam, OpenCV, and RNN

This project uses OpenCV for face detection, captures multiple people with names and gender labels, trains a simple RNN-based model, and recognizes faces with name and gender on webcam input.

## Project Structure

- `dataset/` - stores captured face images organized by `name_gender` folders
- `models/` - stores the trained model and label map
- `capture_faces.py` - capture new faces from webcam and save labeled training images
- `train_model.py` - train the RNN-based recognition model
- `recognize.py` - run webcam face recognition in real time
- `utils.py` - helper functions for face extraction and dataset loading
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add one or more persons:

```bash
python capture_faces.py
```

3. Train the model:

```bash
python train_model.py
```

4. Run recognition:

```bash
python recognize.py
```

## Notes

- Use `c` to capture face images during `capture_faces.py`.
- Use `q` to quit webcam windows.
- The model uses a GRU-based RNN on face image rows to produce identity and gender predictions.
- For better accuracy, capture at least 20 images per person under varying lighting and angles.
