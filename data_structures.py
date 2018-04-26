class Node:
    def __init__(self, snt_index_in_doc=-1,
            start_word_index_in_snt=-1, end_word_index_in_snt=-1,
            node_index_in_doc=-1, node_index_in_snt=-1,
            start_word_index_in_doc=-1, end_word_index_in_doc=-1,
            words='root', label='ROOT'):

        self.snt_index_in_doc = snt_index_in_doc
        self.start_word_index_in_snt = start_word_index_in_snt
        self.end_word_index_in_snt = end_word_index_in_snt
        self.node_index_in_doc = node_index_in_doc
        self.node_index_in_snt = node_index_in_snt
        self.start_word_index_in_doc = start_word_index_in_doc
        self.end_word_index_in_doc = end_word_index_in_doc

        self.words = words
        self.label_full = label
        self.label_timex_event = label.split('-')[0]

        self.parent = None
        self.children = []
        self.ID = '_'.join([str(snt_index_in_doc),
            str(start_word_index_in_snt), str(end_word_index_in_snt)])

    def __str__(self):
        return '\t'.join([self.ID, self.words, self.label])


LABEL_VOCAB_FULL = {
    'ROOT': 0,
    'Timex-AbsoluteConcrete': 1,
    'Timex-RelativeVague': 2,
    'Timex-RelativeConcrete': 3,
    'Event-Event': 4,
    'Event-State': 5,
    'Event-ModalizedEvent': 6,
    'Event-OngoingEvent': 7,
    'Event-GenericHabitual': 8,
    'Event-GenericState': 9,
    'Event-Habitual': 10,
    'Event-CompletedEvent': 11,
    '<UNK>':12,
}

LABEL_VOCAB_TIMEX_EVENT = {
    'ROOT': 0,
    'Timex': 1,
    'Event': 2,
    '<UNK>':3,
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
