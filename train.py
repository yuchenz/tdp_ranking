import sys
import argparse
from data_preparation import make_training_data
from logistic_regression_classifier import LogReg_Classifier
from bilstm_classifier import Bilstm_Classifier


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_file", help="training data")
    arg_parser.add_argument("--timex_event_label_input",
        help="which timex/event label set to use: none, timex_event, or full",
        choices=['none', 'timex_event', 'full'], default='full')
    arg_parser.add_argument("--dev_file", help="dev data")
    arg_parser.add_argument("--model_file", help="where to store the model")
    arg_parser.add_argument("--iter", help="number of interations", type=int)
    arg_parser.add_argument("--labeled", help="train a model to predict labels", 
        action="store_true", default=False)
    arg_parser.add_argument("--classifier", help="which classifier to use",
        choices=["log_reg", "bi_lstm"])
    arg_parser.add_argument("--size_embed", help="word embedding size for bi-lstm model",
        default=32)
    arg_parser.add_argument("--size_timex_event_label_embed",
        help="timex/event label embedding size for bi-lstm model", default=4)
    arg_parser.add_argument("--size_lstm", help="single lstm vector size for bi-lstm model",
        default=32)
    arg_parser.add_argument("--size_hidden",
        help="feed-forward neural network's hidden layer size for bi-lstm model",
        default=32)
    arg_parser.add_argument("--size_edge_label",
        help="number of all possible edge labels", default=11)

    return arg_parser

if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    training_data, vocab = make_training_data(
        args.train_file, args.timex_event_label_input)
    dev_data, _ = make_training_data(
        args.dev_file, args.timex_event_label_input)

    vocab_file = args.model_file + '.vocab'

    if args.classifier == 'log_reg':
        classifier = LogReg_Classifier(vocab)
    elif args.classifier == 'bi_lstm':
        size_edge_label = 1 if not args.labeled else args.size_edge_label
        classifier = Bilstm_Classifier(vocab, args.size_embed, args.size_lstm,
            args.size_hidden, args.timex_event_label_input,
            args.size_timex_event_label_embed, size_edge_label)

    classifier.train(training_data, dev_data, args.model_file, vocab_file, args.labeled, args.iter)
