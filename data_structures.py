import copy


class Node:
    def __init__(self, snt_id=-1, word_id_start=-1, word_id_end=-1, index=-1,
            index_in_snt=-1, word_index_in_doc=-1, words='root', label='ROOT'):
        self.snt_id = snt_id
        self.word_id_start = word_id_start
        self.word_id_end = word_id_end
        self.ID = '_'.join([str(snt_id), str(word_id_start), str(word_id_end)])
        self.index = index
        self.index_in_snt = index_in_snt
        self.word_index_in_doc = word_index_in_doc
        self.words = words
        self.label = label
        self.parent = None
        self.children = []

    def same_span(self, node):
        if node.snt_id == self.snt_id and \
            node.word_id_start == self.word_id_start and \
            node.word_id_end == self.word_id_end:
                return True
        return False

    def __str__(self):
        return '\t'.join([str(self.snt_id), str(self.word_id_start),
                str(self.word_id_end), self.words])


LABEL_VOCAB = {
    'Timex-RelativeConcrete': 0,
    'Timex-AbsoluteConcrete': 1,
    'Timex-RelativeVague': 2,
    'Event': 3,
    'CompletedEvent': 4,
    'State': 5,
    'ModalizedEvent': 6,
    'OngoingEvnet': 7,
    'GenericHabitual': 8,
    'GenericState': 9,
    '<START>': 10,
    '<STOP>': 11,
    '<UNK>': 12,
}
