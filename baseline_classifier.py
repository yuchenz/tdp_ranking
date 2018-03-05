from data_structures import Node


class Baseline_Classifier:
    def __init__(self, default_label):
        self.default_label = default_label

    def predict(self, snt_list, example_list, example, labeled):
        if labeled:
            return self.predict_labeled(snt_list, example)
        else:
            for tup in example:
                tup2 = (tup[0], tup[1], 'EDGE')
                if tup[1].index - tup[0].index == 1:
                    return [(tup2, 1.0)]
            return [((Node(), tup[1], 'EDGE'), 1.0)]

    def predict_labeled(self, snt_list, example):
        for tup in example:
            tup2 = (tup[0], tup[1], self.default_label)
            if tup[1].index - tup[0].index == 1:
                return [(tup2, 1.0)]
        return [((Node(), tup[1], self.default_label), 1.0)]
