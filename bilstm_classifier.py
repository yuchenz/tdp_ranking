import codecs
import random
import json
import os
from vector import Vector
import math
import dynet as dy


class Bilstm_Classifier:
    """Bilstm Classifier. """

    def __init__(self, vocab, size_embed, size_lstm, size_hidden, size_edge_label):
        self.model = dy.Model()

        self.embeddings = self.model.add_lookup_parameters(
                (len(vocab), size_embed))

        self.lstm_fwd = dy.LSTMBuilder(1, size_embed, size_lstm, self.model)
        self.lstm_bwd = dy.LSTMBuilder(1, size_embed, size_lstm, self.model)

        self.pW1 = self.model.add_parameters((size_hidden, 4 * size_lstm))
        self.pb1 = self.model.add_parameters(size_hidden)
        self.pW2 = self.model.add_parameters((size_edge_label, size_hidden))
        self.pb2 = self.model.add_parameters(size_edge_label)

        self.vocab = vocab

    @classmethod
    def load_model(cls, model_file, vocab_file):
        classifier = cls(0, 0, 0, 0, 0)
        classifier.embeddings, classifier.pW1, classifier.pb1, classifier.pW2, \
            classifier.pb2, classifier.lstm_fwd, classifier.lstm_bwd = \
            classifier.model.load(model_file)

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        return classifier

    def train(self, training_data, output_file, vocab_file, labeled, num_iter=1000):
        self.online_train(training_data, output_file, vocab_file, num_iter)

    def get_word_embeddings_for_document(self, snt_list):
        word_list = []
        for snt in snt_list:
            for word in snt:
                word_list.append(word)

        return [dy.lookup(
            self.embeddings,
            self.vocab.get(word, self.vocab['<UNK>'])
            ) for word in word_list]

    def online_train(self, training_data, output_file, vocab_file, num_iter=1000):
        trainer = dy.AdamTrainer(self.model)

        for i in range(num_iter):
            random.shuffle(training_data)
            closs = 0.0
            for snt_list, training_example_list in training_data:
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

                for example in training_example_list:
                    yhat = self.scores(example)
                    loss = self.compute_loss(yhat, self.get_gold_y_index(example))
                    closs += loss.scalar_value()
                    loss.backward()
                    trainer.update()

            if 1:
                print('# iter', i)
                print('loss =', closs)

        self.model.save(
            output_file,
            [self.embeddings, self.pW1, self.pb1, self.pW2, self.pb2,
                self.lstm_fwd, self.lstm_bwd])

        if not os.path.isfile(vocab_file):
            with codecs.open(vocab_file, 'w', 'utf-8') as f:
                json.dump(self.vocab, f)

    def scores(self, example):
        out_list = []
        for tup in example:
            p, c, label = tup
            h = dy.concatenate([self.bi_lstm[p.word_index_in_doc],
                self.bi_lstm[c.word_index_in_doc]])

            hidden = dy.tanh(self.W1 * h + self.b1)
            scores = self.W2 * hidden + self.b2
            out_list.append(scores)

        yhat = dy.concatenate(out_list)
        return yhat

    def get_gold_y_index(self, example):
        out_list = []
        for tup in example:
            p, c, label = tup
            if label != 'NO_EDGE':
                out_list.append(1)
            else:
                out_list.append(0)

        if 1 not in out_list:   # TO DO: investigate, why are there cases with no gold y??
            out_list[0] = 1

        return out_list.index(1) 


    def compute_loss(self, yhat, gold_y_index):
        return -dy.log(dy.pick(dy.softmax(yhat), gold_y_index))
