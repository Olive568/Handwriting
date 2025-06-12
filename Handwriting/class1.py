import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Alphabet and image dimensions
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
img_height, img_width = 64, 256

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret += alphabets[ch]
    return ret

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    final_img = np.ones([img_height, img_width]) * 255
    img = img[:img_height, :img_width]
    final_img[:img.shape[0], :img.shape[1]] = img
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    final_img = final_img / 255.0
    return final_img.reshape(1, img_width, img_height, 1)

def decode_prediction(pred):
    decoded = K.get_value(K.ctc_decode(pred,
                                       input_length=np.ones(pred.shape[0]) * pred.shape[1],
                                       greedy=True)[0][0])
    return num_to_label(decoded[0])

# Load the trained model
model = load_model("Handwriting_model.h5", compile=False)

def test_images_with_metrics(image_dir, test_csv, limit=1000):
    df = pd.read_csv(test_csv)
    df = df[df['IDENTITY'].notnull()]
    df = df[df['IDENTITY'].str.upper() != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper().str.strip()

    predictions = []
    ground_truth = []
    file_paths = []

    for idx, row in tqdm(df.iterrows(), total=min(limit, len(df)), desc="Predicting", unit="img"):
        if idx >= limit:
            break
        file_name = row['FILENAME']
        true_label = row['IDENTITY']
        image_path = os.path.join(image_dir, file_name)

        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is not None:
            pred = model.predict(preprocessed_img, verbose=0)
            pred_label = decode_prediction(pred)
            predictions.append(pred_label)
            ground_truth.append(true_label)
            file_paths.append(file_name)

    # Character-level accuracy
    total_chars = 0
    correct_chars = 0
    exact_match = 0

    for pred, truth in zip(predictions, ground_truth):
        total_chars += len(truth)
        correct_chars += sum(1 for a, b in zip(pred, truth) if a == b)
        if pred == truth:
            exact_match += 1

    acc_char = correct_chars / total_chars
    acc_word = exact_match / len(predictions)

    # For precision and recall, convert exact match to binary
    y_true_bin = [1] * len(predictions)
    y_pred_bin = [1 if p == t else 0 for p, t in zip(predictions, ground_truth)]

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    print("\n--- RESULTS ---")
    print(f"Character-level Accuracy : {acc_char * 100:.2f}%")
    print(f"Word-level Accuracy      : {acc_word * 100:.2f}%")
    print(f"Precision                : {precision * 100:.2f}%")
    print(f"Recall                   : {recall * 100:.2f}%")

    print("\nSample Predictions:")
    for i in range(10):
        print(f"{file_paths[i]} | True: {ground_truth[i]} | Predicted: {predictions[i]}")

# Example usage
if __name__ == "__main__":
    test_images_with_metrics(
        image_dir="C:\\Users\\Luis Oliver\\Datasets\\test_v2\\test",
        test_csv="C:\\Users\\Luis Oliver\\Datasets\\written_name_test_v2.csv",
        limit=1000
    )
