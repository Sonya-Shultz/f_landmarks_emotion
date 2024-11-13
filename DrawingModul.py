import cv2
import numpy as np
from EmotionModul import EmotionRecognition


class DrawingModul:
    def __init__(self):
        self.orig_frame = None
        self.frame = None
        self.landmarks = None
        self.landmarks_raw = None
        self.emotions = None
        self.combined_frame = None
        self.distance_res = None

    def draw(self, orig_frame, landmarks, frame, landmarks_raw, emotions, scale, distance_res):
        self.orig_frame = orig_frame
        self.frame = frame
        self.landmarks = landmarks
        self.landmarks_raw = landmarks_raw
        self.emotions = emotions
        self.distance_res = distance_res
        self.orig_frame = self.__draw_landmarks(self.landmarks, self.orig_frame, scale)
        self.frame = self.__draw_pseudo_landmarks(self.landmarks_raw, self.frame, scale, color=(0, 0, 255))
        self.__combine_frames()
        self.__draw_emotion_section()
        return self.combined_frame

    @staticmethod
    def __draw_landmarks(landmarks_in, frame_in, sf, color=(0, 255, 0)):
        if landmarks_in is not None:
            for landmark, raw_l in zip(landmarks_in, sf):
                scale = max(raw_l[2], raw_l[3])
                for n in range(0, 68):
                    if type(landmark) in [np.ndarray]:
                        x = int(landmark[0, n*2] * scale + raw_l[0])
                        y = int(landmark[0, n*2+1] * scale + raw_l[1])
                    elif type(landmark) in [list]:
                        if type(landmark[n*2]) is not int:
                            x = int(landmark[n*2] * scale + raw_l[0])
                            y = int(landmark[n*2+1] * scale + raw_l[1])
                        else:
                            x = landmark[n*2] + raw_l[0]
                            y = landmark[n*2+1] + raw_l[1]
                    else:
                        x = landmark.part(n).x
                        y = landmark.part(n).y
                    cv2.circle(frame_in, (x, y), 2, color, -1)
        return frame_in

    @staticmethod
    def __draw_pseudo_landmarks(landmarks_in, frame_in, scale, color=(0, 255, 0)):
        if landmarks_in is None:
            return frame_in
        elif type(landmarks_in) in [list] and (len(landmarks_in) > 0 and type(landmarks_in[0]) in [list]):
            for face in landmarks_in:
                if len(face) == 20:
                    cv2.rectangle(frame_in, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (0, 0, 0), 2)
                    for i in range(1, 5):
                        cv2.rectangle(frame_in,
                                      (face[0] + face[i*4], face[1] + face[i*4 + 1]),
                                      (face[0] + face[i*4] + face[i*4+2], face[1] + face[i*4 + 1] + face[i*4+3]),
                                      (0, 50*i, 0), 2)
                else:
                    frame_in = DrawingModul.__draw_landmarks(landmarks_in, frame_in, scale, color)
                    return frame_in
        else:
            frame_in = DrawingModul.__draw_landmarks(landmarks_in, frame_in, scale, color)
        return frame_in

    def __combine_frames(self):
        self.combined_frame = np.hstack((self.frame, self.orig_frame))

    def __draw_emotion_section(self):
        white_rows = np.ones((50, self.combined_frame.shape[1], 3), dtype=np.uint8) * 255
        self.combined_frame = np.vstack((self.combined_frame, white_rows))
        emotion_names = []
        for emotion in self.emotions:
            emotion_names.append(EmotionRecognition.get_top_emotion_name(emotion))

        cv2.putText(
            self.combined_frame,
            ", ".join(emotion_names),
            (50, self.combined_frame.shape[0]-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            self.combined_frame,
            self.distance_res,
            (400, self.combined_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA
        )
