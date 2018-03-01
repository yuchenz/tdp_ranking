import sys
import os
import codecs
import argparse
from data_preparation import make_test_data
from data_preparation import make_test_data_for_bio_tagging
from logistic_regression_classifier import LogReg_Classifier
from baseline_classifier import Baseline_Classifier
from bilstm_classifier import Bilstm_Classifier
from mtl_classifier import MTL_Classifier


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_file", help="test data to be parsed")
    arg_parser.add_argument("--timex_event_label_input",
        help="which timex/event label set to use: none, timex_event, or full",
        choices=['none', 'timex_event', 'full'], default='timex_event')
    arg_parser.add_argument("--model_file", help="the model to use")
    arg_parser.add_argument("--vocab_file", help="the vocab file to use")
    arg_parser.add_argument("--parsed_file", help="where to output the parsed results")
    arg_parser.add_argument("--classifier", help="which classifier to use",
        choices=["log_reg", "bi_lstm", "baseline", "mtl"])
    arg_parser.add_argument("--labeled", help="parse with edge labels",
        action="store_true", default=False)
    arg_parser.add_argument("--default_label",
        help="default edge label to use for baseline parser",
        choices=["before", "overlap"])
    arg_parser.add_argument("--size_embed", help="word embedding size for bi-lstm model",
        default=128)
    arg_parser.add_argument("--size_lstm", help="single lstm vector size for bi-lstm model",
        default=64)
    arg_parser.add_argument("--size_hidden",
        help="feed-forward neural network's hidden layer size for bi-lstm model",
        default=64)
    arg_parser.add_argument("--size_edge_label",
        help="number of all possible edge labels", default=11)

    return arg_parser

def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join([' '.join(snt) for snt in snt_list])
        edge_text = '\n'.join(edge_list)
        f.write('filename:xxx:SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')

def decode(test_data, classifier, output_file, labeled):
    i = 0
    for snt_list, test_instance_list in test_data:
        print('parsing doc {} ...'.format(i))
        i += 1
        edge_list = []
        for instance in test_instance_list:
            yhat_list = classifier.predict(snt_list, test_instance_list,
                instance, labeled)

            yhat = yhat_list[0][0]
            #print([[y[0][0].ID, y[0][1].ID, y[1]] for y in yhat_list])
            #print(yhat)

            edge = '\t'.join([yhat[1].ID, yhat[1].label, yhat[0].ID, yhat[2]])
            edge_list.append(edge)

        output_parse(edge_list, snt_list, output_file)

def convert_bio_to_edge(snt_id, bio_list):
    edge_list = []
    i = 0
    while i < len(bio_list):
        bio = bio_list[i]
        if bio == 'O':
            i += 1
            continue
        else:
            if bio.startswith('B_') or bio.startswith('I_'):
                if bio.startswith('I_'):
                    #print('ERROR!!! O -> I')
                    pass
                start = i
                while i < len(bio_list):
                    if bio_list[i] == 'O':
                        break
                    i += 1
                i -= 1
                end = i
                edge = '_'.join([str(snt_id), str(start), str(end)])
                edge_list.append('\t'.join([edge,
                    bio_list[start].split('_')[-1], '-', '-']))
        i += 1

    return edge_list

def snt_to_doc(snt_list):
    word_list = []
    snt_len_list = []
    for snt in snt_list:
        word_list.extend(snt)
        snt_len_list.append(len(snt))

    return word_list, snt_len_list

def doc_to_snt(bio_doc, snt_len_list):
    bio_list = []
    i = 0
    for snt_len in snt_len_list:
        bio_snt = bio_doc[i:i + snt_len]
        bio_list.append(bio_snt)
        i += snt_len

    return bio_list

def decode_bio(test_data, classifier, output_file):
    for i, snt_list in enumerate(test_data): 
        print('bio tagging doc {} ...'.format(i))

        #import pdb;pdb.set_trace()
        doc, snt_len_list = snt_to_doc(snt_list)
        bio_doc = classifier.bio_predict(doc)
        bio_list = doc_to_snt(bio_doc, snt_len_list)

        edge_list = []
        for j, snt in enumerate(snt_list):
            edge_list.extend(convert_bio_to_edge(j, bio_list[j]))

        output_parse(edge_list, snt_list, output_file)


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    try:
        os.remove(args.parsed_file)
    except OSError:
        pass

    if args.classifier == 'mtl':
        test_data = make_test_data_for_bio_tagging(args.test_file,
            args.timex_event_label_input)

        classifier = MTL_Classifier.load_model(args.model_file,
            args.vocab_file, args.timex_event_label_input)

        if os.path.isfile(args.test_file + '.mtl_bio_tagged'):
            os.remove(args.test_file + '.mtl_bio_tagged')
            open(args.test_file + '.mtl_bio_tagged', 'w').close()

        decode_bio(test_data, classifier, args.test_file + '.mtl_bio_tagged')

        test_data = make_test_data(args.test_file + '.mtl_bio_tagged',
            args.timex_event_label_input)
    else:
        test_data = make_test_data(args.test_file,
            args.timex_event_label_input)

        if args.classifier == 'baseline':
            default_label = args.default_label 
            classifier = Baseline_Classifier(args.default_label)
        elif args.classifier == 'log_reg':
            classifier = LogReg_Classifier.load_model(args.model_file,
                args.vocab_file, args.timex_event_label_input)
        elif args.classifier == 'bi_lstm':
            classifier = Bilstm_Classifier.load_model(args.model_file,
                args.vocab_file, args.timex_event_label_input)

    if os.path.isfile(args.parsed_file):
        os.remove(args.parsed_file)
        open(args.parsed_file, 'w').close()

    decode(test_data, classifier, args.parsed_file, args.labeled)
