import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

class HandwritingRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path, compile=False)
        self.alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        self.img_width = 256
        self.img_height = 64

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h, w) = gray.shape
        canvas = np.ones([self.img_height, self.img_width]) * 255

        # crop or pad
        if w > self.img_width:
            gray = gray[:, :self.img_width]
        if h > self.img_height:
            gray = gray[:self.img_height, :]

        canvas[:gray.shape[0], :gray.shape[1]] = gray
        rotated = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        return rotated / 255.0

    def decode_prediction(self, prediction):
        decoded = K.get_value(K.ctc_decode(prediction, 
                                           input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                           greedy=True)[0][0])
        text = ""
        for ch in decoded[0]:
            if ch == -1:
                break
            text += self.alphabets[ch]
        return text

    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or unreadable.")
        processed = self.preprocess(img)
        pred = self.model.predict(processed.reshape(1, self.img_width, self.img_height, 1))
        return self.decode_prediction(pred)




