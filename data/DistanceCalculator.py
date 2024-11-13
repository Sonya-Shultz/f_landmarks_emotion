class DistanceCalculator:
    @staticmethod
    def calculate_distance(p1_x, p1_y, p2_x, p2_y):
        return ((p1_x-p2_x)**2 + (p1_y-p2_y)**2)**0.5

    @staticmethod
    def __calculate_mean_error(distances, scale, return_message):
        if len(distances) == 0:
            return "--"
        dist = sum(distances)/len(distances)
        if return_message:
            return f"{dist*scale:.3f}px/{dist*100:.3f}%"
        return dist*scale

    @staticmethod
    def calculate_distances(landmarks_1, landmarks_2, scale, return_message=True):
        diff_messages = []
        if len(landmarks_1) != len(landmarks_2):
            return "ERROR: Landmarks have different length!"
        for l1, l2, s in zip(landmarks_1, landmarks_2, scale):
            distances = []
            if len(l1) == len(l2)*2:
                for i in range(0, len(l2)):
                    distances.append(
                        DistanceCalculator.calculate_distance(l1[i * 2], l1[i * 2 + 1], l2[i][0], l2[i][1]))
            elif len(l1)*2 == len(l2):
                for i in range(0, len(l1)):
                    distances.append(
                        DistanceCalculator.calculate_distance(l2[i * 2], l2[i * 2 + 1], l1[i][0], l1[i][1]))
            elif len(l1) == len(l2):
                for i in range(0, len(l1)//2):
                    distances.append(
                        DistanceCalculator.calculate_distance(l2[i * 2], l2[i * 2 + 1], l1[i * 2], l1[i * 2 + 1]))
            else:
                return "ERROR: Landmarks have different length!"
            diff_messages.append(DistanceCalculator.__calculate_mean_error(distances, s, return_message))
        if return_message:
            return f"Mean difference: {', '.join(diff_messages)}"
        return diff_messages
