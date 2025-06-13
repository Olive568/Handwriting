import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras import backend as K

# Define the character set
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

class HandwritingRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.img_width = 256
        self.img_height = 64
        self.num_to_char = {i: ch for i, ch in enumerate(alphabets)}

    def preprocess_image(self, image_path):
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        canvas = np.ones((self.img_height, self.img_width)) * 255  # white canvas

        if w > self.img_width:
            img = img[:, :self.img_width]
        if h > self.img_height:
            img = img[:self.img_height, :]

        canvas[:img.shape[0], :img.shape[1]] = img
        img = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE) / 255.0
        img = img.reshape(1, self.img_width, self.img_height, 1)
        return img

    def decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decoded, log_probs = K.ctc_decode(pred, input_length=input_len, greedy=True)
        decoded = K.get_value(decoded[0])[0]
        prob = np.exp(-K.get_value(log_probs)[0][0])
        result = ''.join([self.num_to_char.get(ch, '') for ch in decoded])
        return result, round(prob * 100, 2)  # confidence in %

    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        pred = self.model.predict(img)
        text, confidence = self.decode_prediction(pred)

        # Show image and prediction
        img_show = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_show, cmap='gray')
        plt.title(f"Prediction: {text} ({confidence}%)", fontsize=12)
        plt.axis('off')
        plt.show()

        return {
            'prediction': text,
            'confidence': confidence
        }

if __name__ == "__main__":
    recognizer = HandwritingRecognizer("Handwriting_model.h5")

    # 🔁 CHANGE THIS TO YOUR TEST IMAGE PATH:
    image_path = "C:\\Users\\Luis Oliver\\Videos\\494861602_723763793673556_6015723790886534530_n.jpg"

    result = recognizer.predict(image_path)

    print("\n📝 Model Output")
    print("Predicted Text :", result['prediction'])
    print("Confidence     :", result['confidence'], "%")
