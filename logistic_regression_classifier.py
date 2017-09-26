import codecs
import json
import os
from vector import Vector
import math


class LogReg_Classifier:
    """ Logistic Regression Classifier. """

    def __init__(self, vocab):
        self.weights = Vector({})
        self.vocab = vocab

    @classmethod
    def load_model(cls, model_file, vocab_file):
        classifier = cls(None)
        classifier.weights = Vector({})
        classifier.weights.load(model_file)

        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            classifier.vocab = json.load(f) 

        return classifier

    def batch_train(self, training_data, output_file, vocab_file, num_iter=1000):
        sum_sq = Vector({})
        for i in range(num_iter):
            grad = Vector({})
            loss = 0
            n = 0
            for snt_list, training_example_list in training_data:
                for example in training_example_list:
                    #yhat_list = self.predict(snt_list, example)
                    #yhat = yhat_list[0][0]

                    l, g = self.compute_loss_and_grad(snt_list, example)
                    if (l, g) == (False, False):    # TO DO: check why this happens
                        break

                    grad += g 
                    loss += l

                    n += 1.0

            grad = (1.0 / n) * grad
            loss /= n

            #if 1: 
            if i % 10 == 0:
                #print('# iter', i, 'weights', self.weights)
                print('# iter', i)
                print('loss =', loss)
                if self.weights.dot(self.weights) != 0:
                    print('normalized loss =', loss / math.sqrt(self.weights.dot(self.weights)))

            # adagrad
            sum_sq += grad.element_wise_square()
            for key in grad.v:
                if grad.v[key] != 0:
                    self.weights.v[key] = \
                        self.weights.v.get(key, 0) - \
                        (0.1 / math.sqrt(sum_sq.v[key])) * grad.v[key]

        self.weights.save(output_file)
        if not os.path.isfile(vocab_file):
            with codecs.open(vocab_file, 'w', 'utf-8') as f:
                json.dump(self.vocab, f)

    def log_sum(self, a, b):
        if a - b > 10:
            return a
        elif b - a > 10:
            return b
        else:
            return a + math.log(1 + math.exp(b - a))

    def compute_loss_and_grad(self, snt_list, example):
        #loss = 0    # TO DO: check why this happens, i.e. no tup in example has edge
        flag = False
        for tup in example:
            if tup[2] != 'NO_EDGE':
                flag = True
                loss = - self.weights.dot(self.extract_feature_vec(snt_list, tup))
                break

        if not flag:        # TO DO: check why this happens
            return False, False

        '''
        norm = 0
        for y in [Action.SHIFT, Action.LEFT_REDUCE,
                Action.RIGHT_REDUCE, Action.ROOT_REDUCE]:
            norm += math.exp(self.weights.dot(
                self.extract_feature_vec(state, y)))
        loss += math.log(norm)
        '''
        scores = []
        log_norm = self.weights.dot(
            self.extract_feature_vec(snt_list, example[0]))
        scores.append(log_norm)
        for tup in example[1:]: 
            tup2 = (tup[0], tup[1], 'EDGE')
            score = self.weights.dot(self.extract_feature_vec(snt_list, tup2))
            log_norm = self.log_sum(log_norm, score)
            scores.append(score)

        loss += log_norm 

        #import pdb; pdb.set_trace()
        #grad = Vector({})   # TO DO: check why this happends, i.e. no tup in example has edge
        for tup in example:
            if tup[2] != 'NO_EDGE':
                grad = -1 * self.extract_feature_vec(snt_list, tup)
                break

        for i, tup in enumerate(example): 
            tup2 = (tup[0], tup[1], 'EDGE')
            prob = math.exp(scores[i] - log_norm)
            grad += prob * self.extract_feature_vec(snt_list, tup2)

        return loss, grad

    def predict(self, snt_list, example):
        yhat = []
        for tup in example: 
            tup2 = (tup[0], tup[1], 'EDGE')
            x = self.extract_feature_vec(snt_list, tup2)
            score = self.weights.dot(x)
            yhat.append((tup2, score))

        return sorted(yhat, key=lambda x: x[1], reverse=True)
    
    def extract_feature_vec(self, snt_list, tup):
        vec = Vector({})

        p_node, c_node, label = tup
        '''
        # p.w, c.w 
        #vec.v['p.w=' + p_node.words] = 1.0
        #vec.v['c.w=' + c_node.words] = 1.0

        # p.l, c.l 
        vec.v['p.l=' + p_node.label] = 1.0
        #vec.v['c.l=' + c_node.label] = 1.0
       
        # pair features
        #vec.v['p.w+c.w=' + p_node.words + '+' + c_node.words] = 1.0
        #vec.v['p.w+c.l=' + p_node.words + '+' + c_node.label] = 1.0
        #vec.v['p.l+c.w=' + p_node.label+ '+' + c_node.words] = 1.0
        #vec.v['p.l+c.l=' + p_node.label+ '+' + c_node.label] = 1.0


        # triple features
        #vec.v['p.w+p.l+c.l=' + p_node.words + '+' + 
        #    p_node.label+ '+' + c_node.label] = 1.0

        #vec.v['wd='] = math.fabs(p_node.word_id_start - c_node.word_id_start) \
        #    if p_node.snt_id == c_node.snt_id else 0

        # Feat: node_distance
        #vec.v['nd='] = math.fabs(c_node.index - p_node.index)
        '''

        # Feat: if p_node is the immediate front node of c_node,
        # i.e. node_distance == 1
        vec.v['nd=1'] = 1.0 if c_node.index - p_node.index == 1 else 0.0
        '''

        # Feat: if p_node is root and c_node is an absolute timex
        if p_node.label == 'ROOT' and c_node.label.startswith('Timex-Absolute'):
            vec.v['p.l_ROOT+c.l_TimexAbsolute=1'] = 1.0
        else:
            vec.v['p.l_ROOT+c.l_TimexAbsolute=1'] = 0.0

        # Feat: if p_node is root and c_node is a timex
        if p_node.label == 'ROOT' and c_node.label.startswith('Timex'):
            vec.v['p.l_ROOT+c.l_Timex=1'] = 1.0
        else:
            vec.v['p.l_ROOT+c.l_Timex=1'] = 0.0

        # Feat: p_node's label & c_node's label
        vec.v['p.l+c.l=' + p_node.label + '+' + c_node.label] = 1.0

        # Feat: if c_node is state, c_node.snt_id != p_node.snt_id
        if c_node.label == 'State' and c_node.snt_id != p_node.snt_id:
            vec.v['c.l_State+pc.nss'] = 1.0
        else:
            vec.v['c.l_State+pc.nss'] = 0.0

        # Feat: if c_node is state, and p_node.index - c_node.index == 1
        if c_node.label == 'State' and p_node.index - c_node.index == 1:
            vec.v['c.l_State+nd=-1'] = 1.0
        else:
            vec.v['c.l_State+nd=-1'] = 0.0

        # Feat: c_node.label and p_node.label (categorized 1)
        vec.v['p.l1+c.l1=' + self.label_categorize_1(p_node.label) + \
            '+' + self.label_categorize_1(c_node.label)] = 1.0

        vec.v['sd='] = math.fabs(p_node.snt_id - c_node.snt_id)

        # Feat: c_node.label and p_node.label (categorized 2)
        vec.v['p.l2+c.l2=' + self.label_categorize_2(p_node.label) + \
            '+' + self.label_categorize_2(c_node.label)] = 1.0

        # Feat: c_node.label and p_node.label (categorized 3)
        vec.v['p.l3+c.l3=' + self.label_categorize_3(p_node.label) + \
            '+' + self.label_categorize_3(c_node.label)] = 1.0

        # p_node c_node in same sentence 
        #vec.v['ss=True' if p_node.snt_id == c_node.snt_id else 'ss=False'] = 1.0
        vec.v['ss=True'] = 1.0 if p_node.snt_id == c_node.snt_id else 0.0

        '''
        # Feat: p_node in quotation marks and c_node in quotation marks
        vec.v['p.q+c.q=' + \
            self.in_quotation_marks(snt_list[p_node.snt_id], p_node) + '+' + \
            self.in_quotation_marks(snt_list[c_node.snt_id], c_node)] = 1.0

        #print('=================vec:', vec)
        return vec 

    def label_categorize_1(self, label):
        if label in 'Event CompletedEvent'.split():
            return 'Event'
        elif label.startswith('Timex'):
            return 'Timex'
        else:
            return 'State'

    def label_categorize_2(self, label):
        if label == 'ROOT':
            return 'ROOT'
        elif label.startswith('Timex'):
            return 'Timex'
        else:
            return 'Event'

    def label_categorize_3(self, label):
        if label == 'ROOT':
            return 'ROOT'
        elif label in 'Event CompletedEvent'.split():
            return 'Event'
        elif label.startswith('Timex'):
            return 'Timex'
        else:
            return 'State'

    def in_quotation_marks(self, snt, node): 
        mode = 'before'
        if "'" in snt or '"' in snt:
            for i in range(len(snt)):
                if mode == 'before' and len(snt[i]) == 1 and ord(snt[i]) in [8220, 8221, 8216, 8217]:
                    mode = 'in'
                if mode == 'in' and i == node.word_id_start:
                    mode = 'in_in'
                if mode == 'in_in' and i == node.word_id_end:
                    mode = 'in_end'
                if mode == 'in_end' and len(snt[i]) == 1 and ord(snt[i]) in [8220, 8221, 8216, 8217]:
                    mode = 'end'
                    break
        if mode == 'end':
            return 'True'
        else:
            return 'False'
