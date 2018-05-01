import sys
import os
import codecs
import argparse
from data_preparation import make_test_data
from logistic_regression_classifier import LogReg_Classifier
from baseline_classifier import Baseline_Classifier
from bilstm_classifier import Bilstm_Classifier


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--test_file", help="test data to be parsed")
    arg_parser.add_argument("--model_file", help="the model to use")
    arg_parser.add_argument("--vocab_file", help="the vocab file to use")
    arg_parser.add_argument("--parsed_file", help="where to output the parsed results")
    
    arg_parser.add_argument("--TE_label_set",
        help="which timex/event label set to use: none, timex_event, or full",
        choices=['none', 'timex_event', 'full'], default='full')

    arg_parser.add_argument("--classifier", help="which classifier to use",
        choices=["log_reg", "bi_lstm", "baseline"])
    arg_parser.add_argument("--labeled", help="parse with edge labels",
        action="store_true", default=False)

    arg_parser.add_argument("--default_label",
        help="default edge label to use for baseline parser",
        choices=["before", "overlap"])

    #arg_parser.add_argument("--size_embed", help="word embedding size for bi-lstm model",
    #    default=16)
    #arg_parser.add_argument("--size_lstm", help="single lstm vector size for bi-lstm model",
    #    default=16)
    #arg_parser.add_argument("--size_hidden",
    #    help="feed-forward neural network's hidden layer size for bi-lstm model",
    #    default=16)
    #arg_parser.add_argument("--size_edge_label",
    #    help="number of all possible edge labels", default=11)

    return arg_parser

def output_parse(edge_list, snt_list, output_file):
    with codecs.open(output_file, 'a', 'utf-8') as f:
        text = '\n'.join([' '.join(snt) for snt in snt_list])
        edge_text = '\n'.join(edge_list)
        f.write(
            'SNT_LIST\n' + text + '\n' + 'EDGE_LIST\n' + edge_text + '\n\n')

def is_cyclic(yhat, offspring_dict):
    candidate = yhat[0].ID
    child = yhat[1].ID

    #print('check candidate', candidate, 'and child', child)

    if child not in offspring_dict:
        #print(child, 'not in offspring_dict')
        return False 
    elif candidate not in offspring_dict[child]:
        #print(candidate, 'not in offspring_dict[', child, ']')
        return False 
    else:
        #print(candidate, 'is in offspring_dict[', child, '], Cyclic!!')
        return True 

def update_dict_w_new_edge(offspring_dict, parent_dict, yhat):
    parent = yhat[0].ID
    child = yhat[1].ID

    #print("\nadd edge: parent", parent, "to child", child)

    parent_dict[child] = parent
    offspring_dict[parent] = offspring_dict.get(parent, {})
    offspring_dict[parent].update({child:None})
    offspring_dict[parent].update(offspring_dict.get(child, {}))
    
    #k = 0
    #while parent in parent_dict and k < 100:
    while parent in parent_dict:
        #k += 1
        #print(parent, end=' ')
        parent = parent_dict[parent]
        offspring_dict[parent] = offspring_dict.get(parent, {})
        offspring_dict[parent].update({child:None})
        offspring_dict[parent].update(offspring_dict.get(child, {}))
    #print(parent, '\n')
    #print("parent_dict:\n", parent_dict)
    #print("offspring_dict:\n", offspring_dict)
    #if k >= 100:
    #    print("ERROR!!!")
    #print()

def decode(test_data, classifier, output_file, labeled, TE_label_set):
    i = 0
    offspring_dict = {}
    parent_dict = {}
    for snt_list, test_instance_list in test_data:
        print('parsing doc {} ...'.format(i))
        i += 1
        edge_list = []
        for instance in test_instance_list:
            yhat_list = classifier.predict(
                snt_list, test_instance_list, instance, labeled)

            # make sure the new edge doesn't add a cycle in the final tree
            j = 0
            while j < len(yhat_list):
                if not is_cyclic(yhat_list[j][0], offspring_dict):
                    break
                j += 1

            assert j < len(yhat_list), "No acyclic edges!!!"

            # update offspring_dict and parent_dict 
            # to include the newly added edge
            update_dict_w_new_edge(
                offspring_dict, parent_dict, yhat_list[j][0])

            yhat = yhat_list[j][0]
            #print([[y[0][0].ID, y[0][1].ID, y[1]] for y in yhat_list])
            #print(yhat)

            if TE_label_set == 'timex_event':
                label = yhat[1].TE_label
            elif TE_label_set == 'full':
                label = yhat[1].full_label
            else:
                label = 'LABEL'

            edge = '\t'.join([yhat[1].ID, label, yhat[0].ID, yhat[2]])
            edge_list.append(edge)


        output_parse(edge_list, snt_list, output_file)


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    try:
        os.remove(args.parsed_file)
    except OSError:
        pass

    test_data = make_test_data(args.test_file)

    if args.classifier == 'baseline':
        default_label = args.default_label 
        classifier = Baseline_Classifier(args.default_label)

    elif args.classifier == 'log_reg':
        classifier = LogReg_Classifier.load_model(
            args.model_file, args.vocab_file)

    elif args.classifier == 'bi_lstm':
        classifier = Bilstm_Classifier.load_model(
             args.model_file, args.vocab_file, args.TE_label_set)

    decode(test_data, classifier, args.parsed_file,
        args.labeled, args.TE_label_set)
