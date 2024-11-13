class DistanceCalculator:
    @staticmethod
    def __calculate_distance(p1_x, p1_y, p2_x, p2_y):
        return ((p1_x-p2_x)**2 + (p1_y-p2_y)**2)**0.5

    @staticmethod
    def __calculate_mean_error(distances, scale):
        if len(distances) == 0:
            return "--"
        dist = sum(distances)/len(distances)
        return f"{dist*scale:.3f}px/{dist*100:.3f}%"

    @staticmethod
    def calculate_distances(landmarks_1, landmarks_2, scale):
        diff_messages = []
        if len(landmarks_1) != len(landmarks_2):
            return "ERROR: Landmarks have different length!"
        for l1, l2, s in zip(landmarks_1, landmarks_2, scale):
            distances = []
            if len(l1) != len(l2)*2:
                return "ERROR: Landmarks have different length!"
            for i in range(0, len(l2)):
                distances.append(DistanceCalculator.__calculate_distance(l1[i*2], l1[i*2+1], l2[i][0], l2[i][1]))
            diff_messages.append(DistanceCalculator.__calculate_mean_error(distances, s))

        return f"Mean difference: {', '.join(diff_messages)}"
