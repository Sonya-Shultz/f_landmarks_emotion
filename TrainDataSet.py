import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from EmotionModul import EmotionRecognition
from FacedetectionModul import Facedetection, LandmarksDetection

# face_detector = Facedetection()
face_detector = LandmarksDetection()
emotion_model = EmotionRecognition()

df = pd.read_csv('training_frames_keypoints.csv')
new_data = []
right_data = []
landmark_columns = df.columns[1:]

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = f"training/{row['image_name']}"
    img = cv2.imread(img_path)

    face_detector.detect(img)
    faces = face_detector.get_faces()
    landmarks = face_detector.get_landmarks()

    if len(faces) == 0:
        continue

    face = face_detector.get_faces()[0]
    landmark = face_detector.get_landmarks()[0]

    emotion_power = emotion_model.get_emotion(img, face)[0]

    row[landmark_columns[::2]] = row[landmark_columns[::2]] - face[0]
    row[landmark_columns[1::2]] = row[landmark_columns[1::2]] - face[1]
    if row[landmark_columns].min() < 0:
        df.drop(index, inplace=True)
        continue

    tmp_landmark = []
    for el in landmark:
        if type(el) is tuple:
            tmp_landmark.append(el[0])
            tmp_landmark.append(el[1])
    df.loc[index, landmark_columns] = row[landmark_columns] / max(face[2], face[3])
    new_data.append([row['image_name']] + tmp_landmark + emotion_power.tolist())
    # new_data.append([row['image_name']] + landmark + emotion_power.tolist())

    # for i, (face, lmks) in enumerate(zip(faces, landmarks)):
    #     emotion_power = emotion_model.get_emotion(img, face)[0]
    #
    #     # lmk_points = [(lmks.part(n).x, lmks.part(n).y) for n in range(68)]
    #     # lmk_flat = np.array(lmk_points).flatten()
    #
    #     new_data.append([row['image_name']] + lmks + emotion_power.tolist())

# columns = ['image_name'] + [f'landmark_{i}' for i in range(5*4)] + [f'emotion_{e}' for e in EmotionRecognition.emotions_order]
columns = ['image_name'] + [f'landmark_{i}' for i in range(68*2)] + [f'emotion_{e}' for e in EmotionRecognition.emotions_order]
new_df = pd.DataFrame(new_data, columns=columns)

new_df.to_csv('processed_landmarks_emotions_v2.csv', index=False)
df.to_csv('result_landmarks_v2.csv', index=False)
print("New dataset saved as 'processed_landmarks_emotions_v2.csv'")
print("New dataset saved as 'result_landmarks_v2.csv'")
