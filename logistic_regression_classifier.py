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
                    yhat_list = self.predict(snt_list, example)
                    yhat = yhat_list[0][0]

                    l, g = self.compute_loss_and_grad(snt_list, example)
                    if (l, g) == (False, False):    # TO DO: check why this happens
                        break

                    grad += g 
                    loss += l

                    n += 1.0

            grad = (1.0 / n) * grad
            loss /= n

            #if i % 100 == 0:
            if 1: 
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
            score = self.weights.dot(self.extract_feature_vec(snt_list, tup))
            log_norm = self.log_sum(log_norm, score)
            scores.append(score)

        loss += log_norm 

        #grad = Vector({})   # TO DO: check why this happends, i.e. no tup in example has edge
        for tup in example:
            if tup[2] != 'NO_EDGE':
                grad = -1 * self.extract_feature_vec(snt_list, tup)
                break

        for i, tup in enumerate(example): 
            prob = math.exp(scores[i] - log_norm)
            grad += prob * self.extract_feature_vec(snt_list, tup)

        return loss, grad

    def predict(self, snt_list, example):
        yhat = []
        for tup in example: 
            x = self.extract_feature_vec(snt_list, tup)
            score = self.weights.dot(x)
            yhat.append((tup, score))

        return sorted(yhat, key=lambda x: x[1], reverse=True)
    
    def extract_feature_vec(self, snt_list, tup):
        vec = Vector({})

        p_node, c_node, label = tup

        '''
        # p.w, c.w 
        vec.v['p.w=' + p_node.words] = 1.0
        vec.v['c.w=' + c_node.words] = 1.0

        # p.l, c.l 
        vec.v['p.l=' + p_node.label] = 1.0
        vec.v['c.l=' + c_node.label] = 1.0
       
        # pair features
        vec.v['p.w+c.w=' + p_node.words + '+' + c_node.words] = 1.0
        vec.v['p.w+c.l=' + p_node.words + '+' + c_node.label] = 1.0
        vec.v['p.l+c.w=' + p_node.label+ '+' + c_node.words] = 1.0
        vec.v['p.l+c.l=' + p_node.label+ '+' + c_node.label] = 1.0

        # triple features
        vec.v['p.w+p.l+c.l=' + p_node.words + '+' + 
            p_node.label+ '+' + c_node.label] = 1.0

        # distance features: same_snt, word_dist, snt_dist
        vec.v['ss=' + 'True' if p_node.snt_id == c_node.snt_id else 'False'] = 1.0

        vec.v['sd='] = math.fabs(p_node.snt_id - c_node.snt_id)

        vec.v['wd='] = math.fabs(p_node.word_id_start - c_node.word_id_start) \
            if p_node.snt_id == c_node.snt_id else 0
        '''

        # if p_node is the immediate front node of c_node,
        # i.e. node_distance == 1
        #vec.v['nd=1'] = 1.0 if c_node.index - p_node.index == 1 else 0.0

        #vec.v['always_on=1'] = 1.0

        #print('=================vec:', vec)
        return vec 
