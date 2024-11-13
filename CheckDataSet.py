import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from EmotionModul import EmotionRecognition
from FacedetectionModul import Facedetection, LandmarksDetection
from LandmarksModul import LandmarksPositioningV2
from CLAHEModul import CLAHE
from data.DistanceCalculator import DistanceCalculator

# face_detector = Facedetection()

face_detector = LandmarksDetection()
emotion_model = EmotionRecognition()
landmarks_model = LandmarksPositioningV2()
CLAHE_modul = CLAHE()

df = pd.read_csv('training_frames_keypoints.csv')
new_data = []
my_data = []
right_data = []
landmark_columns = df.columns[1:]

counter = 0
mae_list = []
mae_full = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    counter += 1
    if counter > 50000:
        break
    img_path = f"training/{row['image_name']}"
    img = cv2.imread(img_path)
    img = CLAHE_modul.apply(img)

    face_detector.detect(img)
    faces = face_detector.get_faces()
    landmarks = face_detector.get_landmarks()

    if len(faces) == 0:
        continue

    emotions = []
    scale = [max(faces[i][2], faces[i][3]) for i in range(len(faces))]
    pos = [[faces[i][0], faces[i][1]] for i in range(len(faces))]
    for face in faces:
        emotion = emotion_model.get_emotion(img, face)
        emotions.append(emotion)
    my_landmarks = landmarks_model.upgrade_landmarks(landmarks, emotions)

    right_landmarks = row[landmark_columns].to_list()
    landmarks = landmarks[0]
    my_landmarks = my_landmarks[0]
    landmarks_n = []
    for el in landmarks:
        if type(el) is tuple:
            landmarks_n.append(round(el[0] * scale[0]) + pos[0][0])
            landmarks_n.append(round(el[1] * scale[0]) + pos[0][1])

    my_landmarks_n = []
    for i in range(len(my_landmarks)):
        my_landmarks_n.append(round(my_landmarks[i] * scale[0] + pos[0][i % 2]))

    my_mae = DistanceCalculator.calculate_distances([right_landmarks], [my_landmarks_n], [1], False)[0]
    lib_mae = DistanceCalculator.calculate_distances([right_landmarks], [landmarks_n], [1], False)[0]

    my_mae_full = [
        DistanceCalculator.calculate_distance(
            right_landmarks[i*2],
            right_landmarks[i*2+1],
            my_landmarks_n[i * 2],
            my_landmarks_n[i * 2 + 1]
        )
        for i in range(len(right_landmarks)//2)
    ]
    lib_mae_full = [
        DistanceCalculator.calculate_distance(
            right_landmarks[i * 2],
            right_landmarks[i * 2 + 1],
            landmarks_n[i * 2],
            landmarks_n[i * 2 + 1]
        )
        for i in range(len(right_landmarks) // 2)
    ]
    # print(f"my_mae: {my_mae}, lib_mae: {lib_mae}")
    mae_list.append((my_mae, lib_mae))
    mae_full.append((my_mae_full, lib_mae_full))

import matplotlib.pyplot as plt
line1_values = [point[0] for point in mae_list]
line2_values = [point[1] for point in mae_list]
diff = [point[0] - point[1] for point in mae_list]
diff_full = [[p1-p2 for p1, p2 in zip(point[0], point[1])] for point in mae_full]
diff_fll_flat = []
for point in diff_full:
    diff_fll_flat.extend(point)
print(len(diff_fll_flat))
time = list(range(1, len(mae_list) + 1))

# plt.plot(time, line1_values, label='my_mae', marker='o')
# plt.plot(time, line2_values, label='lib_mae', marker='o')
# plt.plot(time, diff, label='mae різниця', marker='o')
#
#
# # Add labels and title
# plt.xlabel('No Зображення')
# plt.ylabel('Пікселів')
# plt.title('Різниця середньої абсолютної помилки що до дійсних точок для моєї та бібліотечної реалізацій')
# plt.legend()
# plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(diff_fll_flat, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('MAE різниця')
plt.ylabel('Частота')
plt.title('Гістограма різниці MAE між моєю та бібліотечною реалізацією (що до дійсних точок)')

# Box plot to show dispersion
plt.subplot(1, 2, 2)
plt.boxplot(diff_fll_flat, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.xlabel('MAE різниця')
plt.title('Дисперсія різниці MAE між моєю та бібліотечною реалізацією (що до дійсних точок)')

plt.tight_layout()
plt.show()
