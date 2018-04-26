import codecs
from data_structures import Node


def create_snt_edge_lists(doc):
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

    return snt_list, edge_list

def create_node_list(snt_list, edge_list):
    node_list = []
    snt_node_counter = 0
    for i, edge in enumerate(edge_list):
        child, c_label, parent, l_label = edge
        c_snt, c_start, c_end = [int(ch) for ch in child.split('_')]

        if not c_label.startswith('Timex'):
            c_label = 'Event-' + c_label

        if len(node_list) == 0 or c_snt != node_list[-1].snt_index_in_doc:
            snt_node_counter = 0

        c_node = Node(c_snt, c_start, c_end, i, snt_node_counter,
            get_word_index_in_doc(snt_list, c_snt, c_start),
            get_word_index_in_doc(snt_list, c_snt, c_end),
            ''.join(snt_list[c_snt][c_start:c_end + 1]),
            c_label)

        node_list.append(c_node)
        snt_node_counter += 1

    return node_list

def get_word_index_in_doc(snt_list, snt_index_in_doc, word_index_in_snt):
    index = 0
    for i, snt in enumerate(snt_list):
        if i < snt_index_in_doc:
            index += len(snt)
        else:
            break

    return index + word_index_in_snt

def check_example_contains_gold_parent(example):
    for tup in example:
        if tup[2] != 'NO_EDGE':
            return True

    return False

def make_one_doc_training_data(doc, vocab):
    """
    return: trainining_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'before'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # add words to vocab
    for snt in snt_list:
        for word in snt:
            if word not in vocab:
                vocab[word] = vocab.get(word, 0) + 1

    # create node_list
    node_list = create_node_list(snt_list, edge_list) 

    # create training example list 
    training_example_list = []
    root_node = Node()

    for i, edge in enumerate(edge_list):
        example = []

        _, _, parent, l_label = edge
        c_node = node_list[i]

        if parent == '-1_-1_-1':
            example.append((root_node, c_node, l_label))
        else:
            example.append((root_node, c_node, 'NO_EDGE'))
            
        for candidate_node in node_list:
            if candidate_node.ID == c_node.ID:
                continue
            elif candidate_node.snt_index_in_doc - c_node.snt_index_in_doc > 2:
                break
            else:
                if candidate_node.ID == parent:
                    example.append((candidate_node, c_node, l_label))
                else:
                    example.append((candidate_node, c_node, 'NO_EDGE'))

        assert check_example_contains_gold_parent(example), \
            "ERROR! no gold parent in this example!!!" 

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

    index = 1
    vocab = {}
    for word in count_vocab:
        if count_vocab[word] > 0:
            vocab[word] = index
            index += 1

    vocab.update({'<UNK>':0})

    return training_data, vocab

def make_one_doc_test_data(doc):
    doc = doc.strip().split('\n')

    # create snt_list, edge_list
    snt_list, edge_list = create_snt_edge_lists(doc)

    # create node_list
    node_list = create_node_list(snt_list, edge_list) 
  
    # create test instance list
    test_instance_list = []
    root_node = Node()

    for i, edge in enumerate(edge_list):
        instance = []

        _, _, parent, l_label = edge
        c_node = node_list[i]

        instance.append((root_node, c_node, None))
        
        for candidate_node in node_list:
            if candidate_node.ID == c_node.ID:
                continue
            if candidate_node.snt_index_in_doc - c_node.snt_index_in_doc > 2:
                break
            else:
                instance.append((candidate_node, c_node, None))
        
        test_instance_list.append(instance)

    return [snt_list, test_instance_list]

def make_test_data(test_file):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []
    
    for doc in doc_list:
        test_data.append(make_one_doc_test_data(doc))

    return test_data
