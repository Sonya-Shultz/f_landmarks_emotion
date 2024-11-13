import numpy as np


class Utils:
    @staticmethod
    def fix_pseudo_landmarks_normalization(pl, scale=None):
        new_landmarks = []
        if scale is None or len(scale) != len(pl):
            scale = [1 for _ in range(len(pl))]
        for el, i in zip(pl, scale):
            if type(el) is list:
                if type(el[0]) is tuple:
                    new_el = [0 for _ in range(68 * 2)]
                    for n in range(0, 68):
                        new_el[n * 2] = int(el[n][0] * i)
                        new_el[n * 2 + 1] = int(el[n][1] * i)
                    new_landmarks.append(new_el)
                else:
                    new_el = [0 for _ in range(20)]
                    sf = max(el[2], el[3])
                    for x in range(4):
                        new_el[x] = int(el[x])
                    for x in range(4, len(el)):
                        new_el[x] = int(el[x]*sf)
                    new_landmarks.append(new_el)
            elif type(el) is np.ndarray:
                new_landmarks.append(el)
            else:
                new_el = [0 for _ in range(68*2)]
                for n in range(0, 68):
                    new_el[n*2] = int(el.part(n).x * i)
                    new_el[n*2+1] = int(el.part(n).y * i)
                new_landmarks.append(new_el)

        return new_landmarks
