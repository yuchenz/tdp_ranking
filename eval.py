import codecs
import sys
import argparse


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gold_file", help="gold file")
    arg_parser.add_argument("--parsed_file", help="parsed file")
    arg_parser.add_argument("--labeled", help="evaluate with edge labels",
        action="store_true", default=False)

    return arg_parser

def readin_tuples(filename):
    lines = codecs.open(filename, 'r', 'utf-8').readlines()

    edge_tuples = []
    mode = None
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        elif line.endswith('LIST'):
            mode = line.strip().split(':')[-1]
            if mode == 'SNT_LIST':
                edge_tuples.append([])
        elif mode == 'EDGE_LIST':
            child, child_label, parent, link_label = line.strip().split()
            edge_tuples[-1].append((child, parent, link_label))

    return edge_tuples

def unlabeled_eval(gold_tuples, auto_tuples):
    counts = []
    scores = []

    for i, (gtups, atups) in enumerate(zip(gold_tuples, auto_tuples)):
        gtups = set([(gtup[0], gtup[1]) for gtup in gtups])
        atups = set([(atup[0], atup[1]) for atup in atups])

        true_positive = len(gtups.intersection(atups))
        false_positive = len(atups.difference(gtups))
        false_negative = len(gtups.difference(atups))

        print('test doc {}: true_p = {}, false_p = {}, false_n = {}'.format(
            i, true_positive, false_positive, false_negative)) 
        p = true_positive / (true_positive + false_positive)
        r = true_positive / (true_positive + false_negative)
        f = 2 * p * r / (p + r)

        counts.append((true_positive, false_positive, false_negative))
        scores.append((p, r, f))

    # macro average
    p = sum([score[0] for score in scores]) / len(scores)
    r = sum([score[1] for score in scores]) / len(scores)
    f = sum([score[2] for score in scores]) / len(scores)

    #print('macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    print('macro average: f = {:.3f}'.format(f), end='; ')

    # micro average
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])

    p = true_p / (true_p + false_p)
    r = true_p / (true_p + false_n)
    f = 2 * p * r / (p + r)

    #print('micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    print('micro average: f = {:.3f}'.format(f))

def labeled_eval(gold_tuples, auto_tuples):
    counts = []
    scores = []

    for i, (gtups, atups) in enumerate(zip(gold_tuples, auto_tuples)):
        gtups = set([(gtup[0], gtup[1], gtup[2]) for gtup in gtups])
        atups = set([(atup[0], atup[1], atup[2]) for atup in atups])

        true_positive = len(gtups.intersection(atups))
        false_positive = len(atups.difference(gtups))
        false_negative = len(gtups.difference(atups))

        print('test doc {}: true_p = {}, false_p = {}, false_n = {}'.format(
            i, true_positive, false_positive, false_negative)) 
        p = true_positive / (true_positive + false_positive)
        r = true_positive / (true_positive + false_negative)
        f = 2 * p * r / (p + r) if p + r != 0 else 0

        counts.append((true_positive, false_positive, false_negative))
        scores.append((p, r, f))

    # macro average
    p = sum([score[0] for score in scores]) / len(scores)
    r = sum([score[1] for score in scores]) / len(scores)
    f = sum([score[2] for score in scores]) / len(scores)

    #print('macro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    print('macro average: f = {:.3f}'.format(f), end='; ')

    # micro average
    true_p = sum([count[0] for count in counts])
    false_p = sum([count[1] for count in counts])
    false_n = sum([count[2] for count in counts])

    p = true_p / (true_p + false_p)
    r = true_p / (true_p + false_n)
    f = 2 * p * r / (p + r)

    #print('micro average: p = {:.3f}, r = {:.3f}, f = {:.3f}'.format(p, r, f))
    print('micro average: f = {:.3f}'.format(f))


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    gold_tuples = readin_tuples(args.gold_file)
    auto_tuples = readin_tuples(args.parsed_file)

    if args.labeled:
        labeled_eval(gold_tuples, auto_tuples)
    else:
        unlabeled_eval(gold_tuples, auto_tuples)
