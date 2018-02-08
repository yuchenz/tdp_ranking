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
        self.label = label  # timex/event label
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

def make_label_dict(label_set):
    if label_set == 'BIO':
        return {'B': 0, 'I': 1, 'O': 2}
    elif label_set == 'TIMEX_EVENT':
        return {'B_Timex': 0, 'I_Timex':1, 'B_Event': 2, 'I_Event':3, 'O':4}
    elif label_set == 'FULL':
        return {
            'B_Timex-RelativeConcrete': 0,
            'I_Timex-RelativeConcrete': 1,
            'B_Timex-RelativeVague': 2,
            'I_Timex-RelativeVague': 3,
            'B_Timex-AbsoluteConcrete': 4,
            'I_Timex-AbsoluteConcrete': 5,
            'B_Timex-AbsoluteVague': 6,
            'I_Timex-AbsoluteVague': 7,
            'B_Event': 8,
            'I_Event': 9,
            'B_State': 10,
            'I_State': 11,
            'B_Habitual': 12,
            'I_Habitual': 13,
            'B_ModalizedEvent': 14,
            'I_ModalizedEvent': 15,
            'B_OngoingEvent': 16,
            'I_OngoingEvent': 17,
            'B_GenericState': 18,
            'I_GenericState': 19,
            'B_GenericHabitual': 20,
            'I_GenericHabitual': 21,
            'B_CompletedEvent': 22,
            'I_CompletedEvent': 23,
            'O': 24,
        }
