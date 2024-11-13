from pygrabber.dshow_graph import FilterGraph

from data.DistanceCalculator import DistanceCalculator
from FacedetectionModul import Facedetection, LandmarksDetection
import cv2
from CLAHEModul import CLAHE
from DrawingModul import DrawingModul
from EmotionModul import EmotionRecognition
from LandmarksModul import LandmarksPositioning, LandmarksPositioningV2
from data.Utils import Utils


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

    def capture_loop(self):
        self.cap = cv2.VideoCapture(self.device)
        frame_count = 0

        draw_landmarks_raw = None
        distance_res = ""
        landmarks = None
        faces = None
        emotions = None

        while True:
            ret, frame = self.cap.read()
            orig_frame = frame.copy()
            if not ret:
                break

            frame = self.CLAHE_modul.apply(frame)

            if frame_count % self.image_cap == 0:
                cv2.imwrite(f"images/{frame_count}.png", frame)
                self.facedetect_modul.detect(frame)
                faces = self.facedetect_modul.get_faces()
                landmarks_raw = self.facedetect_modul.get_landmarks()
                scale = [max(faces[i][2], faces[i][3]) for i in range(len(faces))]
                draw_landmarks_raw = Utils.fix_pseudo_landmarks_normalization(landmarks_raw, scale)
                emotions = []
                for face in faces:
                    emotion = self.emotion_modul.get_emotion(frame, face)
                    emotions.append(emotion)
                landmarks = self.landmarks_modul.upgrade_landmarks(landmarks_raw, emotions)
                distance_res = DistanceCalculator.calculate_distances(landmarks, landmarks_raw, scale)
            frame_count = (frame_count + 1) % self.image_cap
            frame = self.drawing_modul.draw(orig_frame, landmarks, frame, draw_landmarks_raw, emotions, faces, distance_res)
            cv2.imshow('Webcam Feed (Left: Processed with CLAHE, Right: Raw)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
