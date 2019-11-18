import codecs
import random
import json
import os
import sys
import math
import operator
import dynet as dy
from vector import Vector
from data_structures import EDGE_LABEL_LIST
from data_structures import LABEL_VOCAB_FULL, LABEL_VOCAB_TIMEX_EVENT 


class Bilstm_Classifier:
    """Bilstm Classifier. """

    def __init__(self, vocab, size_embed, size_lstm, size_hidden,
            timex_event_label_input, size_timex_event_label_embed, 
            size_edge_label=len(EDGE_LABEL_LIST)):

        self.model = dy.Model()
        self.size_edge_label = size_edge_label

        if timex_event_label_input == 'none':
            self.label_vocab = {}
        elif timex_event_label_input == 'timex_event':
            self.label_vocab = LABEL_VOCAB_TIMEX_EVENT
        else:
            self.label_vocab = LABEL_VOCAB_FULL

        if vocab != 0:
            self.embeddings = self.model.add_lookup_parameters(
                    (len(vocab), size_embed))
            self.timex_event_label_embeddings = \
                    self.model.add_lookup_parameters(
                    (len(self.label_vocab), size_timex_event_label_embed))

            self.lstm_fwd = dy.LSTMBuilder(
                    1, size_embed + size_timex_event_label_embed,
                    size_lstm, self.model)
            self.lstm_bwd = dy.LSTMBuilder(
                    1, size_embed + size_timex_event_label_embed,
                    size_lstm, self.model)

            self.pW1 = self.model.add_parameters(
                    (size_hidden, 12 * size_lstm + 5 + 2))
            self.pb1 = self.model.add_parameters(size_hidden)
            self.pW2 = self.model.add_parameters(
                    (size_edge_label, size_hidden))
            self.pb2 = self.model.add_parameters(size_edge_label)

            self.attention_w = self.model.add_parameters((1, size_lstm * 2))

            self.vocab = vocab
            self.size_lstm = size_lstm
        else:
            self.embeddings, self.timex_event_label_embeddings, \
                self.pW1, self.pb1, self.pW2, self.pb2, \
                self.lstm_fwd, self.lstm_bwd, self.attention_w, self.vocab = \
                None, None, None, None, None, None, None, None, None, None

    @classmethod
    def load_model(cls, model_file, vocab_file, timex_event_label_input):
        classifier = cls(0, 0, 0, 0, 0, 0)
        classifier.embeddings, classifier.timex_event_label_embeddings, \
            classifier.pW1, classifier.pb1, classifier.pW2, \
            classifier.pb2, classifier.lstm_fwd, classifier.lstm_bwd,\
            classifier.attention_w = \
            classifier.model.load(model_file)

        classifier.size_edge_label = classifier.pb2.shape()[0]
        if timex_event_label_input == 'none':
            classifier.label_vocab = {}
        elif timex_event_label_input == 'timex_event':
            classifier.label_vocab = LABEL_VOCAB_TIMEX_EVENT
        else:
            classifier.label_vocab = LABEL_VOCAB_FULL

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        classifier.size_lstm = int(classifier.attention_w.shape()[1] / 2)

        return classifier

    def train(self, training_data, dev_data, output_file, vocab_file, labeled,
            bert_train, bert_dev, num_iter=1000):
        self.online_train(training_data, dev_data, output_file, vocab_file, labeled,
                bert_train, bert_dev, num_iter)

    def get_embeddings_for_document(self, snt_list, example_list):
        """ Get word embeddings (concatenated with timex/event label embeddings if picked)
        for bilstm input.
        """

        # build word list
        word_list = []
        timex_event_label_list = []
        for i, snt in enumerate(snt_list):
            for word in snt:
                word_list.append(word)
                timex_event_label_list.append('<UNK>')

        # build timex/event label list
        for example in example_list:
            child = example[0][1]
            for k in range(child.start_word_index_in_doc,
                    child.end_word_index_in_doc + 1):
                timex_event_label_list[k] = child.label

        return [dy.concatenate([
            self.embeddings[self.vocab.get(word, self.vocab['<UNK>'])],
            self.timex_event_label_embeddings[
                self.label_vocab.get(timex_event_label_list[i],
                    self.label_vocab['<UNK>'])]
            ]) for i, word in enumerate(word_list)]

    def load_BERT(bert_filename):
        with open(bert_filename, 'r') as f:
            self.bert_line_list = f.readlines()

    def get_embeddings_BERT(self, snt_list, BERT_line):
        BERT_dict = json.loads(BERT_line)

        BERT_embd_list = []

        char_BERT_list = BERT_dict['features'][1:]
        print('char_BERT_list len:', len(char_BERT_list))
        word_list = [word for snt in snt_list for word in snt]
        print('word_list len:', len(word_list))

        index = 0
        for word in word_list:
            print('word in snt:', word)

            num_char = len(word)
            assert num_char > 0 
            word_char_BERT_list = char_BERT_list[index:index+num_char]
            print(len(char_BERT_list), index, index+num_char, len(word_char_BERT_list))
            index += num_char
            
            char_embd_list = []
            for char_and_layers in word_char_BERT_list:
                print('char in BERT:', char_and_layers['token'])
                layer_1 = list(filter((lambda x: x['index'] == -1), char_and_layers['layers']))[0]

                char_embd_list.append(dy.inputVector(layer_1['values']))

            BERT_embd_list.append(dy.max_dim(dy.concatenate(char_embd_list)))

        return BERT_embd_list

    def build_cg(self, snt_list, example_list, BERT_line):
        dy.renew_cg()

        f_init = self.lstm_fwd.initial_state()
        b_init = self.lstm_bwd.initial_state()

        word_embeddings = self.get_embeddings_for_document(snt_list, example_list)

        BERT_embeddings = self.get_embeddings_BERT(snt_list, BERT_line)

        f_exps = f_init.transduce(word_embeddings)
        b_exps = b_init.transduce(reversed(word_embeddings))

        self.bi_lstm = [dy.concatenate([f, b, BERT]) for f, b, BERT in 
            zip(f_exps, reversed(b_exps), BERT_embeddings)]

        self.W1 = dy.parameter(self.pW1)
        self.b1 = dy.parameter(self.pb1)
        self.W2 = dy.parameter(self.pW2)
        self.b2 = dy.parameter(self.pb2)

    def online_train(self, training_data, dev_data, output_file, vocab_file, labeled,
            bert_train, bert_dev, num_iter=1000):

        trainer = dy.AdamTrainer(self.model)

        dev_loss_inc_count = 0
        pre_dev_loss = 0
        min_dev_loss = sys.maxsize
        for i in range(num_iter):
            training_data_with_BERT = list(zip(training_data, bert_train))
            #random.shuffle(training_data_with_BERT)
            closs = 0.0
            for training_doc, BERT_line in training_data_with_BERT:
                snt_list, training_example_list = training_doc
                #import pdb; pdb.set_trace()
                self.build_cg(snt_list, training_example_list, BERT_line)

                for example in training_example_list:
                    yhat = self.scores(example)
                    loss = self.compute_loss(yhat, self.get_gold_y_index(example, labeled))
                    closs += loss.scalar_value()
                    loss.backward()
                    trainer.update()

            # compute dev loss for early stopping
            dev_loss = 0.0
            dev_data_with_BERT = list(zip(dev_data, bert_dev))
            for dev_doc, BERT_line in dev_data_with_BERT:
                snt_list, dev_example_list = dev_doc
                self.build_cg(snt_list, dev_example_list, BERT_line)
                for example in dev_example_list:
                    yhat = self.scores(example)
                    loss = self.compute_loss(yhat, self.get_gold_y_index(example, labeled))
                    dev_loss += loss.scalar_value()

            print('# iter', i, end=': ')
            print('loss on training = {}, loss on dev = {}'.format(closs, dev_loss))

            # early stopping
            if dev_loss > pre_dev_loss:
                dev_loss_inc_count += 1
            else:
                dev_loss_inc_count = 0
            # if 3 consecutive iters have increasing dev loss, then break
            if dev_loss_inc_count > 2:  
                break
            else:
                pre_dev_loss = dev_loss

            # if dev loss decreased, save model
            if dev_loss < min_dev_loss:
                self.model.save(
                    output_file,
                    [self.embeddings, self.timex_event_label_embeddings,
                        self.pW1, self.pb1, self.pW2, self.pb2,
                        self.lstm_fwd, self.lstm_bwd, self.attention_w])

            min_dev_loss = min(min_dev_loss, dev_loss)

        with codecs.open(vocab_file, 'w', 'utf-8') as f:
            json.dump(self.vocab, f)

    def feat_nd_bin(self, p, c):
        vec = [0 for i in range(5)]
        if c.index - p.index == 1:
            vec[0] = 1
        elif c.index - p.index > 1 and c.snt_id == p.snt_id:
            vec[1] = 1
        elif c.index - p.index > 1:
            vec[2] = 1
        elif c.index - p.index < 1:
            vec[3] = 1
        else:
            vec[4] = 1

        return vec

    def scores(self, example):
        out_list = []
        for tup in example:
            p, c, label = tup

            # feat: node distance 
            nd = dy.inputVector(self.feat_nd_bin(p, c))

            # feat: in same sentence
            ss = dy.inputVector([1, 0] if p.snt_id == c.snt_id else [0, 1])

            # pair representation g
            g = dy.concatenate([
                self.bi_lstm[p.start_word_index_in_doc],
                self.bi_lstm[p.end_word_index_in_doc],
                self.bi_lstm[c.start_word_index_in_doc],
                self.bi_lstm[c.end_word_index_in_doc],
                self.attend(p), self.attend(c), nd, ss])

            hidden = dy.tanh(self.W1 * g + self.b1)
            scores = self.W2 * hidden + self.b2
            out_list.append(scores)

        yhat = dy.concatenate(out_list)

        # for debug
        #print('unnormalized predicted y vector len:', len(yhat.value()))
        #guess = sorted(enumerate(yhat.npvalue()),
        #    key=operator.itemgetter(1), reverse=True)[0][0]
        #guess = self.yhat_index_to_yhat(guess, example)
        #print('predicted:\t', guess[0], '\t', guess[1], '\t', guess[2])
        # end debug

        return yhat

    def attend(self, node):
        '''attention mechanism to return a weighted sum of bilstm vectors
        of all words in node '''

        if node.snt_id == -1:   # if node is a pre-defined meta node
            return self.bi_lstm[node.start_word_index_in_doc]

        #print(node.start_word_index_in_doc, node.end_word_index_in_doc)
        vectors = self.bi_lstm[
                node.start_word_index_in_doc:node.end_word_index_in_doc + 1]

        # build attention on a larger context
        # +/- n words around the current timex/event
        # n = 2
        if node.start_word_index_in_doc <= 0:
            vectors.insert(0, 
                dy.inputVector([0 for i in range(self.size_lstm * 2)]))
        else:
            vectors.insert(0, self.bi_lstm[node.start_word_index_in_doc - 1])

        if node.start_word_index_in_doc <= 1:
            vectors.insert(0,
                dy.inputVector([0 for i in range(self.size_lstm * 2)]))
        else:
            vectors.insert(0, self.bi_lstm[node.start_word_index_in_doc - 2])

        if node.end_word_index_in_doc >= len(self.bi_lstm) - 1:
            vectors.append(
                dy.inputVector([0 for i in range(self.size_lstm * 2)]))
        else:
            vectors.append(self.bi_lstm[node.end_word_index_in_doc + 1])

        if node.end_word_index_in_doc >= len(self.bi_lstm) - 2:
            vectors.append(
                dy.inputVector([0 for i in range(self.size_lstm * 2)]))
            #print(node.start_word_index_in_doc, node.end_word_index_in_doc)
            #print(len(self.bi_lstm))
            #print(self.size_lstm)
        else:
            vectors.append(self.bi_lstm[node.end_word_index_in_doc + 2])

        input_mat = dy.concatenate_cols(vectors)

        attn_w = dy.parameter(self.attention_w)

        unnormalized = dy.transpose(dy.tanh(attn_w * input_mat))
        att_weights = dy.softmax(unnormalized)

        weighted_sum = input_mat * att_weights

        return weighted_sum

    def get_gold_y_index(self, example, labeled):
        out_list = []
        for tup in example:
            p, c, label = tup
            if labeled:
                for l in EDGE_LABEL_LIST:
                    if l == label:
                        out_list.append(1)
                    else:
                        out_list.append(0)
            elif not labeled:
                if label != 'NO_EDGE':
                    out_list.append(1)
                else:
                    out_list.append(0)

        # for debug
        #print('gold y vector:', out_list)
        #print('gold y vector len:', len(out_list))
        #yhat = self.yhat_index_to_yhat(out_list.index(1), example)
        #print('gold answer:\t', yhat[0], '\t', yhat[1], '\t', yhat[2])
        # end debug

        return out_list.index(1) 

    def compute_loss(self, yhat, gold_y_index):
        return -dy.log(dy.pick(dy.softmax(yhat), gold_y_index))

    def yhat_index_to_yhat(self, yhat_index, example, labeled):
        example_index = int(yhat_index / self.size_edge_label)
        label_index = yhat_index % self.size_edge_label

        label = EDGE_LABEL_LIST[label_index] if labeled else 'EDGE'
        tup = example[example_index]
        yhat = (tup[0], tup[1], label)

        return yhat

    def predict(self, snt_list, example_list, example, labeled, BERT_line):
        self.build_cg(snt_list, example_list, BERT_line)
        yhat_list = sorted(enumerate(self.scores(example).npvalue()),
            key=operator.itemgetter(1), reverse=True)

        yhat = self.yhat_index_to_yhat(yhat_list[0][0], example, labeled)

        # for debug
        #print('predicted:\t', yhat[0], '\t', yhat[1], '\t', yhat[2])
        # end debug

        return [(yhat, 0)]

    def label_one_hot(self, label):
        vec = [0 for key in self.self.label_vocab]
        vec[self.label_vocab[label]] = 1
        return dy.inputVector(vec)
