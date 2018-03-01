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
from data_structures import make_label_dict


class MTL_Classifier:
    """Bilstm Classifier. """

    def __init__(self, vocab, size_embed, size_lstm, size_tag_hidden,
            size_parse_hidden, timex_event_label_input, 
            size_edge_label=len(EDGE_LABEL_LIST)):
        self.model = dy.Model()
        self.size_edge_label = size_edge_label

        if timex_event_label_input == 'none':
            self.label_vocab = {}
            self.bio_label_dict = make_label_dict('BIO')
        elif timex_event_label_input == 'timex_event':
            self.label_vocab = LABEL_VOCAB_TIMEX_EVENT
            self.bio_label_dict = make_label_dict('TIMEX_EVENT')
        else:
            self.label_vocab = LABEL_VOCAB_FULL
            self.bio_label_dict = make_label_dict('FULL')
        size_bio_label = len(self.bio_label_dict)

        if vocab != 0:
            self.embeddings = self.model.add_lookup_parameters(
                (len(vocab), size_embed))

            self.lstm_fwd = dy.LSTMBuilder(
                1, size_embed, size_lstm, self.model)
            self.lstm_bwd = dy.LSTMBuilder(
                1, size_embed, size_lstm, self.model)

            self.pW1 = self.model.add_parameters(
                (size_tag_hidden, 2 * size_lstm))
            self.pb1 = self.model.add_parameters(size_tag_hidden)
            self.pW2 = self.model.add_parameters(
                (size_bio_label, size_tag_hidden))
            self.pb2 = self.model.add_parameters(size_bio_label)

            self.pW3 = self.model.add_parameters(
                (size_parse_hidden, 4 * size_lstm))
            self.pb3 = self.model.add_parameters(size_parse_hidden)
            self.pW4 = self.model.add_parameters(
                (size_edge_label, size_parse_hidden))
            self.pb4 = self.model.add_parameters(size_edge_label)

            self.vocab = vocab
        else:
            self.embeddings, self.pW1, self.pb1, self.pW2, self.pb2, \
                self.pW3, self.pb3, self.pW4, self.pb4, \
                self.lstm_fwd, self.lstm_bwd, self.vocab = None, None, None, \
                None, None, None, None, None, None, None, None, None

    @classmethod
    def load_model(cls, model_file, vocab_file, timex_event_label_input):
        classifier = cls(0, 0, 0, 0, 0, 0)
        classifier.embeddings, \
            classifier.pW1, classifier.pb1, classifier.pW2, classifier.pb2, \
            classifier.pW3, classifier.pb3, classifier.pW4, classifier.pb4, \
            classifier.lstm_fwd, classifier.lstm_bwd = \
            classifier.model.load(model_file)

        classifier.size_edge_label = classifier.pb2.shape()[0]
        if timex_event_label_input == 'none':
            classifier.label_vocab = {}
            classifier.bio_label_dict = make_label_dict('BIO')
        elif timex_event_label_input == 'timex_event':
            classifier.label_vocab = LABEL_VOCAB_TIMEX_EVENT
            classifier.bio_label_dict = make_label_dict('TIMEX_EVENT')
        else:
            classifier.label_vocab = LABEL_VOCAB_FULL
            classifier.bio_label_dict = make_label_dict('FULL')

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        return classifier

    def train(self, training_data, dev_data, output_file, vocab_file,
            labeled, num_iter=1000):
        self.online_train(training_data, dev_data, output_file, vocab_file, labeled, num_iter)

    def get_embeddings_for_document(self, snt_list, example_list):
        """ Get word embeddings for bilstm input.
        """

        # build word list
        word_list = []
        for i, snt in enumerate(snt_list):
            for word in snt:
                word_list.append(word)

        return [self.embeddings[self.vocab.get(word, self.vocab['<UNK>'])]
            for i, word in enumerate(word_list)]

    def build_cg(self, snt_list, example_list):
        dy.renew_cg()

        f_init = self.lstm_fwd.initial_state()
        b_init = self.lstm_bwd.initial_state()

        embeddings = self.get_embeddings_for_document(snt_list, example_list)

        f_exps = f_init.transduce(embeddings)
        b_exps = b_init.transduce(reversed(embeddings))

        self.bi_lstm = [dy.concatenate([f, b]) for f, b in 
            zip(f_exps, reversed(b_exps))]

        self.W1 = dy.parameter(self.pW1)
        self.b1 = dy.parameter(self.pb1)
        self.W2 = dy.parameter(self.pW2)
        self.b2 = dy.parameter(self.pb2)

        self.W3 = dy.parameter(self.pW3)
        self.b3 = dy.parameter(self.pb3)
        self.W4 = dy.parameter(self.pW4)
        self.b4 = dy.parameter(self.pb4)

    def online_train(self, training_data, dev_data, output_file, vocab_file,
            labeled, num_iter=1000):
        trainer = dy.AdamTrainer(self.model)

        dev_loss_inc_count = 0
        pre_dev_loss = 0
        min_dev_loss = sys.maxsize
        for i in range(num_iter):
            random.shuffle(training_data)
            closs = 0.0
            for snt_list, training_example_list, gold_bio_list \
                    in training_data:
                #import pdb; pdb.set_trace()
                self.build_cg(snt_list, training_example_list)

                loss = None

                # loss of all timex/event in the doc
                for example in training_example_list:
                    parse_yhat = self.parse_scores(example)
                    parse_loss = self.compute_loss(parse_yhat,
                        self.get_gold_y_index(example, labeled))

                    if not loss:
                        loss = parse_loss
                    else:
                        loss += parse_loss
                    closs += parse_loss.scalar_value()

                word_list = [word for snt in snt_list for word in snt]
                gold_bio_word_list = [bio
                    for snt in gold_bio_list for bio in snt]
                # loss of all words in the doc
                for j, word in enumerate(word_list):
                    tag_yhat = self.tag_scores(j)
                    tag_loss = self.compute_loss(tag_yhat,
                        self.bio_label_dict[gold_bio_word_list[j]])

                    loss += tag_loss
                    closs += tag_loss.scalar_value()

                # update weight for each doc, online training?
                loss.backward()     
                trainer.update()

            # compute dev loss for early stopping
            dev_loss = 0.0
            for snt_list, dev_example_list, gold_bio_list in dev_data:
                self.build_cg(snt_list, dev_example_list)

                # loss of all timex/event in the doc
                for example in dev_example_list:
                    parse_yhat = self.parse_scores(example)
                    parse_loss = self.compute_loss(parse_yhat,
                        self.get_gold_y_index(example, labeled))

                    dev_loss += parse_loss.scalar_value()

                word_list = [word for snt in snt_list for word in snt]
                gold_bio_word_list = [bio
                    for snt in gold_bio_list for bio in snt]
                # loss of all words in the doc
                for j, word in enumerate(word_list):
                    tag_yhat = self.tag_scores(j)
                    tag_loss = self.compute_loss(tag_yhat,
                        self.bio_label_dict[gold_bio_word_list[j]])

                    dev_loss += tag_loss.scalar_value()

            print('# iter', i, end=': ')
            print('loss on training = {}, loss on dev = {}'.format(
                closs / len(training_data),
                dev_loss / len(dev_data)))

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
                    [self.embeddings,
                        self.pW1, self.pb1, self.pW2, self.pb2,
                        self.pW3, self.pb3, self.pW4, self.pb4,
                        self.lstm_fwd, self.lstm_bwd])

            min_dev_loss = min(min_dev_loss, dev_loss)

        with codecs.open(vocab_file, 'w', 'utf-8') as f:
            json.dump(self.vocab, f)

    def parse_scores(self, example):
        out_list = []
        for tup in example:
            p, c, label = tup

            h = dy.concatenate([self.bi_lstm[p.word_index_in_doc],
                self.bi_lstm[c.word_index_in_doc]])

            hidden = dy.tanh(self.W3 * h + self.b3)
            scores = self.W4 * hidden + self.b4
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

    def tag_scores(self, word_index_in_doc):
        h = self.bi_lstm[word_index_in_doc]

        hidden = dy.tanh(self.W1 * h + self.b1)
        yhat = self.W2 * hidden + self.b2

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

    def predict(self, snt_list, example_list, example, labeled):
        self.build_cg(snt_list, example_list)
        yhat_list = sorted(enumerate(self.parse_scores(example).npvalue()),
            key=operator.itemgetter(1), reverse=True)

        yhat = self.yhat_index_to_yhat(yhat_list[0][0], example, labeled)

        # for debug
        #print('predicted:\t', yhat[0], '\t', yhat[1], '\t', yhat[2])
        # end debug

        return [(yhat, 0)]

    def bio_predict(self, snt):
        # build_cg
        dy.renew_cg()

        f_init = self.lstm_fwd.initial_state()
        b_init = self.lstm_bwd.initial_state()

        embeddings = [self.embeddings[
            self.vocab.get(word, self.vocab['<UNK>'])]
            for word in snt] 

        f_exps = f_init.transduce(embeddings)
        b_exps = f_init.transduce(reversed(embeddings))

        self.bi_lstm = [dy.concatenate([f, b]) for f, b in
            zip(f_exps, reversed(b_exps))]

        self.W1 = dy.parameter(self.pW1)
        self.b1 = dy.parameter(self.pb1)
        self.W2 = dy.parameter(self.pW2)
        self.b2 = dy.parameter(self.pb2)

        # predict
        bio_list = []
        for i, word in enumerate(snt):
            yhat_list = sorted(enumerate(self.tag_scores(i).npvalue()),
                key=operator.itemgetter(1), reverse=True)

            id2bio = {self.bio_label_dict[bio]:bio
                for bio in self.bio_label_dict}

            yhat = id2bio[yhat_list[0][0]]
            bio_list.append(yhat)

        return bio_list
