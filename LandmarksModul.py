import numpy as np
import pandas as pd
from keras.models import load_model


class LandmarksPositioning:
    def __init__(self):
        self.model = load_model("LandmarksModel/LandmarksModel.h5")

    def upgrade_landmarks(self, landmarks, emotions):
        new_landmarks = []
        for emotion, landmark in zip(emotions, landmarks):
            lmk_pd = pd.array(landmark).reshape(-1, 20)
            landmark_data = self.model.predict([lmk_pd, emotion], verbose=0)
            new_landmarks.append(landmark_data)

        return new_landmarks


class LandmarksPositioningV2:
    def __init__(self):
        self.model = load_model("LandmarksModel/LandmarksModel_v3.h5")
        pass

    @staticmethod
    def __combine_landmarks(old_landmarks, new_landmarks, sf=0.2):  # 0.3 for v2
        combined_landmarks = []
        for j in range(len(old_landmarks)):
            combined_landmark = [0 for _ in range(len(old_landmarks[j])*2)]
            for i in range(len(old_landmarks[j])):
                combined_landmark[i * 2] = (1 - sf) * old_landmarks[j][i][0] + sf * new_landmarks[j][i * 2]
                combined_landmark[i * 2 + 1] = (1 - sf) * old_landmarks[j][i][1] + sf * new_landmarks[j][i * 2 + 1]
            combined_landmarks.append(combined_landmark)
        return combined_landmarks

    def upgrade_landmarks(self, landmarks, emotions):
        new_landmarks = []
        for emotion, landmark in zip(emotions, landmarks):
            landmark_data = [0 for _ in range(len(landmark)*2)]
            for i in range(len(landmark)):
                landmark_data[i*2] = landmark[i][0]
                landmark_data[i*2+1] = landmark[i][1]
            # new_landmarks.append(landmark_data)
            lmk_pd = pd.array(landmark_data).reshape(-1, 68*2)
            landmark_data = self.model.predict([lmk_pd, emotion], verbose=0)
            new_landmarks.append(landmark_data[0])
        if len(new_landmarks) == 0:
            return new_landmarks
        return LandmarksPositioningV2.__combine_landmarks(landmarks, new_landmarks)

