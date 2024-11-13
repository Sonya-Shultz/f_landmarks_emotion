import cv2


class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_frame = self.clahe.apply(gray)
        return cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)
