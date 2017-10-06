import sys
import os
import codecs
from data_preparation import make_test_data
from logistic_regression_classifier import LogReg_Classifier
from baseline_classifier import Baseline_Classifier


def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join([' '.join(snt) for snt in snt_list])
        edge_text = '\n'.join(edge_list)
        f.write('SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')

def decode(test_data, classifier, output_file, labeled):
    i = 0
    for snt_list, test_instance_list in test_data:
        print('parsing doc {} ...'.format(i))
        i += 1
        edge_list = []
        for instance in test_instance_list:
            if labeled:
                yhat_list = classifier.predict_labeled(snt_list, instance)
            else:
                yhat_list = classifier.predict(snt_list, instance)

            yhat = yhat_list[0][0]
            #print([[y[0][0].ID, y[0][1].ID, y[1]] for y in yhat_list])
            #print(yhat)

            edge = '\t'.join([yhat[1].ID, yhat[1].label, yhat[0].ID, yhat[2]])
            edge_list.append(edge)

        output_parse(edge_list, snt_list, output_file)


if __name__ == '__main__':
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    vocab_file = sys.argv[3]
    parsed_file = sys.argv[4]
    clas = sys.argv[5]
    labeled = True if sys.argv[6] == 'labeled' else False

    try:
        os.remove(parsed_file)
    except OSError:
        pass

    test_data = make_test_data(test_file)

    if clas == 'baseline':
        default_label = sys.argv[7]
        classifier = Baseline_Classifier(default_label)
    else:
        classifier = LogReg_Classifier.load_model(model_file, vocab_file)

    decode(test_data, classifier, parsed_file, labeled)
