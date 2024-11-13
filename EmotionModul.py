import cv2
import numpy as np
from keras.models import load_model


class EmotionRecognition:
    emotions_order = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def __init__(self):
        self.model = load_model('EmotionModel/PhotoModel.h5')

    @staticmethod
    def get_top_emotion_name(prediction):
        return EmotionRecognition.emotions_order[np.argmax(prediction)]

    def get_emotion(self, img, position):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = position
        face_gray = gray[y:y + h, x:x + w]
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray_np = np.reshape(face_gray, (48, 48))
        face_gray_np = np.array([face_gray_np])
        prediction = self.model.predict(face_gray_np, verbose=0)
        return prediction
