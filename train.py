import sys
from data_preparation import make_training_data
from logistic_regression_classifier import LogReg_Classifier


if __name__ == '__main__':
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    num_iter = int(sys.argv[3])
    labeled = True if sys.argv[4] == 'labeled' else False

    training_data, vocab = make_training_data(train_file)

    classifier = LogReg_Classifier(vocab)

    vocab_file = 'models/' + \
        '_'.join(train_file.split('/')[-1].split('.')) + '.vocab'

    classifier.batch_train(training_data, model_file, vocab_file, labeled, num_iter)
