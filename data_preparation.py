import codecs
import pdb
from data_structures import Node


def get_word_index_in_doc(snt_list, snt_id, word_id):
    index = 0
    for i, snt in enumerate(snt_list):
        if i < snt_id:
            index += len(snt)
        else:
            break

    return index + word_id


def check_example_contains_1(example):
    for tup in example:
        if tup[2] != 'NO_EDGE':
            return True

    return False


def make_one_doc_training_data(doc, vocab, timex_event_label_input):
    """
    return: trainining_example_list
    [[(p_node, c_node, 'NO_EDGE'), (p_node, c_node, 'before'), ...],
        [(...), (...), ...],
        ...]
    """

    doc = doc.strip().split('\n')

    # create snt_list, initial gold_bio_list (only 'O's), edge_list
    snt_list = []
    gold_bio_list = []
    edge_list = []
    mode = None
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
            gold_bio_list.append(['O' for word in snt_list[-1]])
        elif mode == 'EDGE_LIST':
            edge_list.append(line.strip().split())

    # add to vocab
    for snt in snt_list:
        for word in snt:
            if word not in vocab:
                vocab[word] = vocab.get(word, 0) + 1

    # create node_list, update gold_bio_list
    node_list = []
    snt_node_counter = 0
    for i, edge in enumerate(edge_list):
        child, c_label, parent, l_label = edge
        if timex_event_label_input == 'timex_event':
            if c_label.startswith('Timex'):
                c_label = "Timex"
            else:
                c_label = "Event"
        elif timex_event_label_input == 'none':
            c_label = 'none'
            
        c_snt, c_start, c_end = child.split('_')

        # update gold_bio_list
        gold_bio_list[int(c_snt)][int(c_start)] = 'B_' + c_label
        for j in range(int(c_start) + 1, int(c_end) + 1):
            gold_bio_list[int(c_snt)][int(j)] = 'I_' + c_label

        if len(node_list) == 0 or c_snt != node_list[-1].snt_id:
            snt_node_counter = 0

        c_node = Node(int(c_snt), int(c_start), int(c_end), i, snt_node_counter,
            get_word_index_in_doc(snt_list, int(c_snt), int(c_start)),
            ''.join(snt_list[int(c_snt)][int(c_start):int(c_end) + 1]),
            c_label)
        node_list.append(c_node)

        snt_node_counter += 1

    # create training example list 
    training_example_list = []
    root_node = Node()

    for i, edge in enumerate(edge_list):
        example = []

        child, c_label, parent, l_label = edge
        p_snt, p_start, p_end = parent.split('_') 
        c_snt, c_start, c_end = child.split('_')
        c_node = node_list[i]

        if parent == '-1_-1_-1':
            example.append((root_node, c_node, l_label))
        else:
            example.append((root_node, c_node, 'NO_EDGE'))
        #pdb.set_trace()
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

        if not check_example_contains_1(example):
            print('ERROR! no gold parent in this example!!!')
            #pdb.set_trace()
            exit(1)

        training_example_list.append(example)

    return [snt_list, training_example_list, gold_bio_list]


def make_training_data(train_file, timex_event_label_input):
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
        training_data.append(make_one_doc_training_data(
            doc, count_vocab, timex_event_label_input))

    index = 3
    vocab = {}
    for word in count_vocab:
        if count_vocab[word] > 0:
            vocab[word] = index
            index += 1

    vocab.update({'<START>':0, '<STOP>':1, '<UNK>':2})

    return training_data, vocab


def make_one_doc_test_data(doc, timex_event_label_input):
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
    snt_node_counter = 0
    for i, edge in enumerate(edge_list):
        child, c_label, parent, l_label = edge
        if timex_event_label_input == 'timex_event':
            if c_label.startswith('Timex'):
                c_label = "Timex"
            else:
                c_label = "Event"
        elif timex_event_label_input == 'none':
            c_label = 'none'

        c_snt, c_start, c_end = child.split('_')

        if len(node_list) == 0 or c_snt != node_list[-1].snt_id:
            snt_node_counter = 0

        c_node = Node(int(c_snt), int(c_start), int(c_end), i, snt_node_counter, 
            get_word_index_in_doc(snt_list, int(c_snt), int(c_start)),
            ''.join(snt_list[int(c_snt)][int(c_start):int(c_end) + 1]),
            c_label)
        node_list.append(c_node)

        snt_node_counter += 1

    # create test instance list
    test_instance_list = []
    root_node = Node()

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


def make_test_data(test_file, timex_event_label_input):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []
    for doc in doc_list:
        test_data.append(make_one_doc_test_data(doc, timex_event_label_input))

    return test_data


def make_one_doc_test_data_for_bio_tagging(doc, timex_event_label_input):
    doc = doc.strip().split('\n')

    # create snt_list
    snt_list = []
    for line in doc:
        if line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
        elif mode == 'SNT_LIST':
            snt_list.append(line.strip().split())
        elif mode == 'EDGE_LIST':
            break

    return snt_list


def make_test_data_for_bio_tagging(test_file, timex_event_label_input):
    data = codecs.open(test_file, 'r', 'utf-8').read()
    doc_list = data.strip().split('\n\nfilename')

    test_data = []
    for doc in doc_list:
        test_data.append(make_one_doc_test_data_for_bio_tagging(
            doc, timex_event_label_input))

    return test_data
