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
from data_structures import LABEL_VOCAB 


class Bilstm_Classifier:
    """Bilstm Classifier. """

    def __init__(self, vocab, size_embed, size_lstm, size_hidden,
            size_edge_label=len(EDGE_LABEL_LIST)):
        self.model = dy.Model()
        self.size_edge_label = size_edge_label

        if vocab != 0:
            self.embeddings = self.model.add_lookup_parameters(
                    (len(vocab), size_embed))

            self.lstm_fwd = dy.LSTMBuilder(1, size_embed, size_lstm, self.model)
            self.lstm_bwd = dy.LSTMBuilder(1, size_embed, size_lstm, self.model)

            # option1: use only bi-lstm vectors in the input layer of the ffnn
            #self.pW1 = self.model.add_parameters(
            #    (size_hidden, 4 * size_lstm))

            # option2: add gold label one hot vectors in the input layer of the ffnn
            self.pW1 = self.model.add_parameters(
                (size_hidden, 4 * size_lstm + 2 * len(LABEL_VOCAB)))

            self.pb1 = self.model.add_parameters(size_hidden)
            self.pW2 = self.model.add_parameters((size_edge_label, size_hidden))
            self.pb2 = self.model.add_parameters(size_edge_label)

            self.vocab = vocab
        else:
            self.embeddings, self.pW1, self.pb1, self.pW2, self.pb2, \
                self.lstm_fwd, self.lstm_bwd, self.vocab = None, None, None, \
                None, None, None, None, None

    @classmethod
    def load_model(cls, model_file, vocab_file):
        classifier = cls(0, 0, 0, 0)
        classifier.embeddings, classifier.pW1, classifier.pb1, classifier.pW2, \
            classifier.pb2, classifier.lstm_fwd, classifier.lstm_bwd = \
            classifier.model.load(model_file)

        classifier.size_edge_label = classifier.pb2.shape()[0]

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        return classifier

    def train(self, training_data, dev_data, output_file, vocab_file, labeled, num_iter=1000):
        self.online_train(training_data, dev_data, output_file, vocab_file, labeled, num_iter)

    def get_word_embeddings_for_document(self, snt_list):
        word_list = []
        for snt in snt_list:
            for word in snt:
                word_list.append(word)

        return [dy.lookup(
            self.embeddings,
            self.vocab.get(word, self.vocab['<UNK>'])
            ) for word in word_list]

    def build_cg(self, snt_list):
        dy.renew_cg()

        f_init = self.lstm_fwd.initial_state()
        b_init = self.lstm_bwd.initial_state()

        word_embeddings = self.get_word_embeddings_for_document(snt_list)

        f_exps = f_init.transduce(word_embeddings)
        b_exps = b_init.transduce(reversed(word_embeddings))

        self.bi_lstm = [dy.concatenate([f, b]) for f, b in 
            zip(f_exps, reversed(b_exps))]

        self.W1 = dy.parameter(self.pW1)
        self.b1 = dy.parameter(self.pb1)
        self.W2 = dy.parameter(self.pW2)
        self.b2 = dy.parameter(self.pb2)

    def online_train(self, training_data, dev_data, output_file, vocab_file, labeled, num_iter=1000):
        trainer = dy.AdamTrainer(self.model)

        dev_loss_inc_count = 0
        pre_dev_loss = 0
        min_dev_loss = sys.maxsize
        for i in range(num_iter):
            random.shuffle(training_data)
            closs = 0.0
            for snt_list, training_example_list in training_data:
                self.build_cg(snt_list)

                for example in training_example_list:
                    yhat = self.scores(example)
                    loss = self.compute_loss(yhat, self.get_gold_y_index(example, labeled))
                    closs += loss.scalar_value()
                    loss.backward()
                    trainer.update()

            # compute dev loss for early stopping
            dev_loss = 0.0
            for snt_list, dev_example_list in dev_data:
                self.build_cg(snt_list)
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
            if dev_loss_inc_count > 3:  
                break
            else:
                pre_dev_loss = dev_loss

            # if dev loss decreased, save model
            if dev_loss < min_dev_loss:
                self.model.save(
                    output_file,
                    [self.embeddings, self.pW1, self.pb1, self.pW2, self.pb2,
                        self.lstm_fwd, self.lstm_bwd])

            min_dev_loss = min(min_dev_loss, dev_loss)

        with codecs.open(vocab_file, 'w', 'utf-8') as f:
            json.dump(self.vocab, f)

    def scores(self, example):
        out_list = []
        for tup in example:
            p, c, label = tup

            # option1: use only bi-lstm vectors in the input layer of the ffnn
            #h = dy.concatenate([self.bi_lstm[p.word_index_in_doc],
            #    self.bi_lstm[c.word_index_in_doc]])

            # option2: add gold label one hot vectors in the input layer of the ffnn
            h = dy.concatenate([self.bi_lstm[p.word_index_in_doc],
                self.bi_lstm[c.word_index_in_doc],
                #self.label_one_hot(p.label)])
                self.label_one_hot(p.label),
                self.label_one_hot(c.label)])

            hidden = dy.tanh(self.W1 * h + self.b1)
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

    def predict(self, snt_list, example, labeled):
        self.build_cg(snt_list)
        yhat_list = sorted(enumerate(self.scores(example).npvalue()),
            key=operator.itemgetter(1), reverse=True)

        yhat = self.yhat_index_to_yhat(yhat_list[0][0], example, labeled)

        # for debug
        #print('predicted:\t', yhat[0], '\t', yhat[1], '\t', yhat[2])
        # end debug

        return [(yhat, 0)]

    def label_one_hot(self, label):
        vec = [0 for key in LABEL_VOCAB]
        vec[LABEL_VOCAB[label]] = 1
        return dy.inputVector(vec)
