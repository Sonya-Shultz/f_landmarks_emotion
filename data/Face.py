class Face:

    def __init__(self, bounds, eyes, nose, mouth):
        self.bounds = bounds
        self.l_eye = Face.__get_side_eye(eyes)
        self.r_eye = Face.__get_side_eye(eyes, r=True)
        self.nose = nose
        self.mouth = mouth
        self.__fix_positions()

    @staticmethod
    def __fix_one_position(pos, sf):
        if len(pos) < 4:
            return [-1, -1, -1, -1]
        return [x/sf for x in pos]

    def __fix_positions(self):
        if len(self.bounds) < 4:
            self.bounds = [-1, -1, -1, -1]

        scale_factor = max(self.bounds[2],self.bounds[3])
        self.l_eye = self.__fix_one_position(self.l_eye, scale_factor)
        self.r_eye = self.__fix_one_position(self.r_eye, scale_factor)
        self.nose = self.__fix_one_position(self.nose, scale_factor)
        self.mouth = self.__fix_one_position(self.mouth, scale_factor)

    @staticmethod
    def __get_side_eye(eyes, r=False):
        if len(eyes) < 8:
            return eyes[:4]
        if r:
            if eyes[0] > eyes[4]:
                return eyes[:4]
            else:
                return eyes[4:]
        else:
            if eyes[0] > eyes[4]:
                return eyes[4:]
            else:
                return eyes[:4]

    def flatten(self):
        tmp = []
        tmp.extend(self.bounds)
        tmp.extend(self.mouth)
        tmp.extend(self.nose)
        tmp.extend(self.r_eye)
        tmp.extend(self.l_eye)
        return tmp
