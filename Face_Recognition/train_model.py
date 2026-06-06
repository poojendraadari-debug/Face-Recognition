import os
import json
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import ensure_directories, load_dataset, save_label_map


MODEL_PATH = os.path.join("models", "face_recognition_rnn.h5")
LABEL_PATH = os.path.join("models", "label_map.json")


def build_rnn_model(input_shape, num_identities):
    inputs = Input(shape=input_shape, name="face_input")
    x = Reshape((input_shape[0], input_shape[1]))(inputs)
    x = GRU(128, return_sequences=False)(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    identity_output = Dense(num_identities, activation="softmax", name="identity_output")(x)
    gender_output = Dense(1, activation="sigmoid", name="gender_output")(x)
    model = Model(inputs=inputs, outputs=[identity_output, gender_output], name="face_rnn_model")
    model.compile(
        optimizer=Adam(learning_rate=0.0007),
        loss={"identity_output": "sparse_categorical_crossentropy", "gender_output": "binary_crossentropy"},
        metrics={"identity_output": "accuracy", "gender_output": "accuracy"},
    )
    return model


def train():
    ensure_directories()
    X, y_name, y_gender, name_to_label, label_to_name = load_dataset("dataset")

    num_identities = len(name_to_label)
    if num_identities < 1:
        raise ValueError("Add at least one person to the dataset before training.")

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_index = int(len(X) * 0.85)
    train_idx = indices[:split_index]
    val_idx = indices[split_index:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train_name, y_val_name = y_name[train_idx], y_name[val_idx]
    y_train_gender, y_val_gender = y_gender[train_idx], y_gender[val_idx]

    model = build_rnn_model((64, 64, 1), num_identities)
    print(model.summary())

    model.fit(
        X_train,
        {"identity_output": y_train_name, "gender_output": y_train_gender},
        validation_data=(X_val, {"identity_output": y_val_name, "gender_output": y_val_gender}),
        epochs=25,
        batch_size=16,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    save_label_map({"label_to_name": label_to_name}, LABEL_PATH)
    print(f"Saved label map to {LABEL_PATH}")


if __name__ == "__main__":
    train()
