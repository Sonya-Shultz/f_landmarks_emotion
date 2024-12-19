from pygrabber.dshow_graph import FilterGraph

from data.DistanceCalculator import DistanceCalculator
from FacedetectionModul import Facedetection, LandmarksDetection
import cv2
from CLAHEModul import CLAHE
from DrawingModul import DrawingModul
from EmotionModul import EmotionRecognition
from LandmarksModul import LandmarksPositioning, LandmarksPositioningV2
from data.Utils import Utils
from time import time
import numpy as np


class ImageCapture:
    def __init__(self, image_cap=1):
        graph = FilterGraph()
        try:
            self.device = graph.get_input_devices().index("HP TrueVision HD Camera")
        except ValueError:
            self.device = 0
        self.cap = None
        self.CLAHE_modul = CLAHE()
        self.facedetect_modul = Facedetection()
        self.facedetect_modul = LandmarksDetection()
        self.emotion_modul = EmotionRecognition()
        self.landmarks_modul = LandmarksPositioning()
        self.landmarks_modul = LandmarksPositioningV2()
        self.drawing_modul = DrawingModul()
        self.image_cap = image_cap

        self.fpses = []

    def capture_loop(self):
        self.cap = cv2.VideoCapture(self.device)
        frame_count = 0

        draw_landmarks_raw = None
        distance_res = ""
        landmarks = None
        faces = None
        emotions = None

        while True:
            start = time()
            ret, frame = self.cap.read()
            orig_frame = frame.copy()
            if not ret:
                break

            frame = self.CLAHE_modul.apply(frame)
            # cv2.imwrite(f"images/{frame_count}.png", frame)
            self.facedetect_modul.detect(frame)
            faces = self.facedetect_modul.get_faces()
            landmarks_raw = self.facedetect_modul.get_landmarks()
            scale = [max(faces[i][2], faces[i][3]) for i in range(len(faces))]
            draw_landmarks_raw = Utils.fix_pseudo_landmarks_normalization(landmarks_raw, scale)
            emotions_t = []
            for face, i in zip(faces, range(len(faces))):
                if frame_count % self.image_cap == 0:
                    emotion = self.emotion_modul.get_emotion(frame, face)
                else:
                    try:
                        emotion = emotions[i]
                    except IndexError:
                        emotion = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                emotions_t.append(emotion)

            emotions = emotions_t
            if len(faces) > 0:
                landmarks = self.landmarks_modul.upgrade_landmarks(landmarks_raw, emotions)
                distance_res = DistanceCalculator.calculate_distances(landmarks, landmarks_raw, scale)
            frame_count = (frame_count + 1) % self.image_cap
            frame = self.drawing_modul.draw(orig_frame, landmarks, frame, draw_landmarks_raw, emotions, faces, distance_res)
            cv2.imshow('Webcam Feed (Left: Processed with CLAHE, Right: Raw)', frame)
            fps = 1.0/(time()-start)
            self.calc_mean_fps(fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def calc_mean_fps(self, fp, cap=60):
        self.fpses.append(fp)
        if len(self.fpses) < cap:
            return
        elif len(self.fpses) == cap:
            print(f"MEAN FPS: {sum(self.fpses) / len(self.fpses)}")
            self.fpses = []
        else:
            self.fpses = []
