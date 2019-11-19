import argparse
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import json

parser = argparse.ArgumentParser()

parser.add_argument('--input_file')
parser.add_argument('--prediction_file')
parser.add_argument('--do_lower_case', action='store_true')
parser.add_argument('--exclude_stop_words', action='store_true')
parser.add_argument('--ratio', type=float, default=0.02)
parser.add_argument('--num_bins', type=int, default=10)

args = parser.parse_args()


with open(args.input_file, 'r') as f:
    data = f.readlines()

stop_words = set(stopwords.words('english'))

def pre_process_sent(sent):

    if args.do_lower_case:
        sent = sent.lower()

    sent.replace('"', '')
    tokens = sent.split()

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    new_sent = []
    for token in tokens:
        if token not in symbols and len(token) > 1:
            new_sent.append(token)

    final_sent = []
    if args.exclude_stop_words:
        for token in new_sent:
            if token not in stop_words:
                final_sent.append(token)
    else:
        final_sent = new_sent

    return final_sent

N = len(data)

doc_turns = []
sentences = []

for val in data[:max(1, int(N * args.ratio))]:
    name, sent = val.split('\t', 1)
    doc_name, turn = name.rsplit('/', 1)
    if doc_name.startswith('/'):
        doc_name = doc_name[1:]
    doc_turns.append([doc_name, turn])

    final_sent = pre_process_sent(sent)
    sentences.append(final_sent)

pass


df = {}
for i, (k, t) in enumerate(doc_turns):
    tokens = sentences[i]
    for w in tokens:
        if w not in df.keys():
            df[w] = {i}
        else:
            df[w].add(i)


tf_idf = {}
for i, (k, t) in enumerate(doc_turns):
    tokens = sentences[i]
    counter = Counter(tokens)
    for token in np.unique(tokens):
        termf = counter[token] / len(tokens)
        docf = len(df[token])
        idf = np.log(N/(docf+1))
        tf_idf[k, t, token] = termf*idf

pass


with open(args.prediction_file) as fp:
    pred_data = json.load(fp)
def remove_none_slots(belief):
    for slot_tuple in belief:
        domain, slot_name, slot_value = slot_tuple.split('-')
        if slot_value == 'none':
            continue
        yield slot_tuple
def get_joint_accuracy(turn):
    return float(set(remove_none_slots(turn['turn_belief'])) == set(remove_none_slots(turn['pred_bs_ptr'])))


accuracies = {}
for i, (k, t) in enumerate(doc_turns):
    for turn in pred_data[k].keys():
        acc = get_joint_accuracy(pred_data[k][turn])
        accuracies[k, turn] = acc
pass


keys = set([x for x in accuracies.keys()])
doc_keys = set([tuple(x) for x in doc_turns])

new_tf_idf = {}
for i, (k, t) in enumerate(doc_turns):
    new_tf_idf[k, t] = []
    tokens = sentences[i]
    for word in tokens:
        new_tf_idf[k, t].append(tf_idf[k, t, word])

for i, (k, t) in enumerate(doc_turns):
    new_tf_idf[k, t] = sum(new_tf_idf[k, t]) / len(new_tf_idf[k, t])


min_val, max_val = min(new_tf_idf.values()), max(new_tf_idf.values())

num_bins = args.num_bins
step = (max_val - min_val) / num_bins

bins = [(i*step, (i+1)*step) for i in range(num_bins)]

bins_acc = {}

for i, bin in enumerate(bins):
    bins_acc[i] = []
    for j, (k, t) in enumerate(doc_turns):
            try:
                if new_tf_idf[k, t] >= bin[0] and new_tf_idf[k, t] < bin[1]:
                    bins_acc[i].append(accuracies[k, t])
            except:
                print('*******')

for i in range(len(bins)):
    if len(bins_acc[i]) == 0:
        bins_acc[i] = 0.0
    else:
        bins_acc[i] = sum(bins_acc[i]) / len(bins_acc[i])


plt.plot(list(bins_acc.values()))
plt.show()








