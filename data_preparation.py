import codecs
import pdb
from data_structures import Node


def make_one_doc_training_data(doc, vocab):
    """
    return: trainining_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'before'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    # add to vocab
    for snt in snt_list:
        for word in snt:
            if word not in vocab:
                vocab[word] = vocab.get(word, 0) + 1

    # create node_list
    node_list = []
    for i, edge in enumerate(edge_list):
        child, c_label, parent, l_label = edge
        c_snt, c_start, c_end = child.split('_')
        c_node = Node(int(c_snt), int(c_start), int(c_end), i,
            ''.join(snt_list[int(c_snt)][int(c_start):int(c_end) + 1]),
            c_label)
        node_list.append(c_node)

    # create training example list 
    training_example_list = []
    root_node = Node(-1, -1, -1, -1)

    for i, edge in enumerate(edge_list):
        example = []

        child, c_label, parent, l_label = edge
        p_snt, p_start, p_end = parent.split('_') 
        c_node = node_list[i]

        if parent == '-1_-1_-1':
            example.append((root_node, c_node, l_label))
        else:
            example.append((root_node, c_node, 'NO_EDGE'))
        for p_node in node_list:
            if p_node.ID == child:
                continue
            elif p_node.snt_id - int(c_snt) > 2:
                break
            else:
                if p_node.ID == parent:
                    example.append((p_node, c_node, l_label))
                else:
                    example.append((p_node, c_node, 'NO_EDGE'))

        training_example_list.append(example)

    return [snt_list, training_example_list]


def make_training_data(train_file):
    """ Given a file of multiple documents in conll-similar format,
    produce a list of training docs, each training doc is 
    (1) a list of sentences in that document; and 
    (2) a list of (parent_candidate, child_node, edge_label/no_edge) tuples 
    in that document; 
    and the vocabulary of this training data set.
    """

    data = codecs.open(train_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    training_data = []
    count_vocab = {}

    for doc in doc_list:
        training_data.append(make_one_doc_training_data(doc, count_vocab))

    index = 3
    vocab = {}
    for word in count_vocab:
        if count_vocab[word] > 0:
            vocab[word] = index
            index += 1

    vocab.update({'<START>':0, '<STOP>':1, '<UNK>':2})

    return training_data, vocab


def make_one_doc_test_data(doc):
    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    # create node_list
    node_list = []
    for i, edge in enumerate(edge_list):
        child, c_label, parent, l_label = edge
        c_snt, c_start, c_end = child.split('_')
        c_node = Node(int(c_snt), int(c_start), int(c_end), i,
                ''.join(snt_list[int(c_snt)][int(c_start):int(c_end) + 1]),
            c_label)
        node_list.append(c_node)

    # create test instance list
    test_instance_list = []
    root_node = Node(-1, -1, -1, -1)

    for i, edge in enumerate(edge_list):
        instance = []

        child, c_label, parent, l_label = edge
        c_snt, c_start, c_end = child.split('_')

        c_node = node_list[i]
        instance.append((root_node, c_node, 'EDGE'))
        
        for p_node in node_list:
            if p_node.ID == child:
                continue
            if p_node.snt_id - int(c_snt) > 2:
                break
            else:
                instance.append((p_node, c_node, 'EDGE'))
        
        test_instance_list.append(instance)

    return [snt_list, test_instance_list]


def make_test_data(test_file):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []
    
    for doc in doc_list:
        test_data.append(make_one_doc_test_data(doc))

    return test_data
