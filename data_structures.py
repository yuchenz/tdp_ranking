import copy


class Node:
    def __init__(self, snt_id=-1, word_id_start=-1, word_id_end=-1, index=-1,
            index_in_snt=-1, start_word_index_in_doc=-1,
            end_word_index_in_doc=-1, words='root', label='ROOT'):
        self.snt_id = snt_id
        self.word_id_start = word_id_start
        self.word_id_end = word_id_end
        self.ID = '_'.join([str(snt_id), str(word_id_start), str(word_id_end)])
        self.index = index
        self.index_in_snt = index_in_snt
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc
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
                str(self.word_id_end), self.words, self.label])


LABEL_VOCAB_FULL = {
    'Timex-RelativeConcrete': 0,
    'Timex-AbsoluteConcrete': 1,
    'Timex-RelativeVague': 2,
    'Event': 3,
    'CompletedEvent': 4,
    'State': 5,
    'ModalizedEvent': 6,
    'OngoingEvent': 7,
    'GenericHabitual': 8,
    'GenericState': 9,
    '<START>': 10,
    '<STOP>': 11,
    '<UNK>': 12,
    'ROOT': 13,
    'Habitual': 14,
}

LABEL_VOCAB_TIMEX_EVENT = {
    'Timex': 0,
    'Event': 1,
    '<START>': 2,
    '<STOP>': 3,
    '<UNK>': 4,
    'ROOT': 5,
}

EDGE_LABEL_LIST = [
    'ROOT',
    'DCT',
    'PRESENT_REF',
    'PAST_REF',
    'FUTURE_REF',
    'ATEMPORAL',
    'before',
    'after',
    'overlap',
    'includes',
    'Depend-on',
]
