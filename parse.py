import sys
import codecs
from data_preparation import make_training_data
from logistic_regression_classifier import LogReg_Classifier


def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join([' '.join(snt) for snt in snt_list])
        edge_text = '\n'.join(edge_list)
        f.write('SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')

def decode(test_data, classifier, output_file):
    i = 0
    for snt_list, training_example_list in test_data:
        print('parsing doc {} ...'.format(i))
        i += 1
        edge_list = []
        for example in training_example_list:
            yhat_list = classifier.predict(snt_list, example)
            yhat = yhat_list[0][0]

            edge = '\t'.join([yhat[1].ID, yhat[1].label, yhat[0].ID, 'link'])
            edge_list.append(edge)

        output_parse(edge_list, snt_list, output_file)


if __name__ == '__main__':
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    vocab_file = sys.argv[3]
    parsed_file = sys.argv[4]

    test_data, vocab = make_training_data(test_file)

    classifier = LogReg_Classifier.load_model(model_file, vocab_file)

    decode(test_data, classifier, parsed_file)
