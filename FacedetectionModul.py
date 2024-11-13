import cv2
import dlib
from data.Face import Face


class Facedetection:
    def __init__(self):
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.__nose_cascade = cv2.CascadeClassifier('./LandmarksModel/cv2cascades/haarcascade_mcs_nose.xml')
        self.__eyes_cascade = cv2.CascadeClassifier('./LandmarksModel/cv2cascades/haarcascade_eye.xml')
        self.__mouth_cascade = cv2.CascadeClassifier('./LandmarksModel/cv2cascades/haarcascade_mcs_mouth.xml')
        self._faces = None
        self._pseudo_landmarks = None

    def get_faces(self):
        return self._faces

    def get_landmarks(self):
        return self._pseudo_landmarks

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)

        self._pseudo_landmarks = []

        for (x, y, w, h) in self._faces:

            roi_gray = gray[y:y + h, x:x + w]

            eyes = self.__eyes_cascade.detectMultiScale(roi_gray, 1.1, 4)[:2]
            eyes_pos = []
            for (ex, ey, ew, eh) in eyes:
                eyes_pos.extend([ex, ey, ew, eh])

            nose = self.__nose_cascade.detectMultiScale(roi_gray, 1.1, 4)[:1]
            nose_pos = []
            for (nx, ny, nw, nh) in nose:
                nose_pos.extend([nx, ny, nw, nh])

            mouth = self.__mouth_cascade.detectMultiScale(roi_gray, 1.1, 4)[:1]
            mouth_pos = []
            for (mx, my, mw, mh) in mouth:
                mouth_pos.extend([mx, my, mw, mh])

            tmp_face = Face([x, y, w, h], eyes_pos, nose_pos, mouth_pos)
            self._pseudo_landmarks.append(tmp_face.flatten())


class LandmarksDetection:
    def __init__(self):
        self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.__predictor = dlib.shape_predictor("./LandmarksModel/shape_predictor_68_face_landmarks.dat")
        self.__detector = dlib.get_frontal_face_detector()
        self._faces = None
        self._pseudo_landmarks = None

    def get_faces(self):
        return self._faces

    def get_landmarks(self):
        return self._pseudo_landmarks

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)

        self._pseudo_landmarks = []

        for (x, y, w, h) in self._faces:
            scale = max(w, h)
            roi_gray = gray[y:y + h, x:x + w]
            rect = dlib.rectangle(int(0), int(0), int(w), int(h))
            landmarks = self.__predictor(roi_gray, rect)
            landmarks_new = []
            for p in landmarks.parts():
                p_x = p.x/scale
                p_y = p.y/scale
                landmarks_new.append((p_x, p_y))
            self._pseudo_landmarks.append(landmarks_new)



