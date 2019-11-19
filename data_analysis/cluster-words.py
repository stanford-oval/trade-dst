#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
import numpy as np
import re

from utils.fix_label import fix_general_label_error

EXPERIMENT_DOMAINS = ["none", "hotel", "train", "restaurant", "attraction", "taxi"]
DOMAIN_INDICES = dict()
for domain in EXPERIMENT_DOMAINS:
    DOMAIN_INDICES[domain] = len(DOMAIN_INDICES)
def get_slot_information():
    ontology = json.load(open("data/multi-woz/MULTIWOZ2.1/ontology.json", 'r'))
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS
ALL_SLOTS = get_slot_information()


def replace_with(sentence, string, replacement):
    string_words = string.split()

    i = 0
    while i < len(sentence):
        found = True
        for j in range(len(string_words)):
            if i+j >= len(sentence) or sentence[i+j] != string_words[j]:
                found = False
                break
        if found:
            yield replacement
            i += len(string_words)
        else:
            yield sentence[i]
            i += 1


def preprocess(sentence):
    sentence = sentence.split()

    for word in sentence:
        if word in ('.', '?', ',', '!'):
            continue

        if re.match('[0-9]{,2}:[0-9]{2}$', word, re.IGNORECASE):
            yield 'TIME'
        elif re.match('\$?[0-9]+$', word, re.IGNORECASE):
            if len(word) >= 9:
                yield 'PHONE_NUMBER'
            else:
                yield 'NUMBER'
        elif re.match('#[0-9a-z]+$', word, re.IGNORECASE):
            yield 'RESERVATION_CODE'
        elif re.match('tr[0-9]+$', word, re.IGNORECASE):
            yield 'TRAIN_NUMBER'
        elif re.match('cb[0-9a-z]{4,}$', word, re.IGNORECASE):
            yield 'POST_CODE'
        elif re.match('([a-z]*[0-9]+[a-z]*)+$', word, re.IGNORECASE): # word with number
            yield 'CONFIRMATION_NUMBER'
        else:
            yield word


def replace_with_system_act(sentence, system_acts):
    sentence = sentence.split()
    for act in system_acts:
        if isinstance(act, str):
            continue

        assert isinstance(act, (list, tuple))
        key, value = act
        if value in ('yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SYSTEM_' + key))

    return ' '.join(sentence)


def replace_with_slots(sentence, turn_label, label_dict):
    sentence = sentence.split()
    for slot in turn_label:
        assert isinstance(slot, (list, tuple))
        key, _ = slot
        value = label_dict[key]
        if value in ('yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SLOT_' + re.sub('[ -]', '_', key)))

    return ' '.join(sentence)


def load_data():
    filename = 'data/train_dials.json'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    with open(filename) as fp:
        data = json.load(fp)

    for dialogue in data:
        dialogue_idx = dialogue['dialogue_idx']

        dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

        prev_turn = 'none'

        for turn in dialogue['dialogue']:
            system_transcript = ' '.join(preprocess(turn['system_transcript']))
            transcript = ' '.join(preprocess(turn['transcript']))

            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)

            if prev_turn in EXPERIMENT_DOMAINS and system_transcript.strip() != '':
                yield replace_with_system_act(system_transcript, turn['system_acts']), prev_turn, 'system'

            if turn['domain'] in EXPERIMENT_DOMAINS:
                yield replace_with_slots(transcript, turn['turn_label'], label_dict), turn['domain'], 'user'

            prev_turn = turn['domain']

PRINT_CLUSTERS = False
PRINT_TEMPLATES = True

def supervised_cluster():
    all_words = defaultdict(lambda: np.zeros((len(EXPERIMENT_DOMAINS),)))

    data = list(load_data())

    # compute the distribution of domains for each word
    for sentence, domain, direction in data:
        words = sentence.split()
        for word in words:
            if word == ';':
                continue
            all_words[word][DOMAIN_INDICES[domain]] += 1

    clusters = defaultdict(set)

    # normalize and cluster
    for word, distribution in all_words.items():
        # normalize to a prob distribution
        distribution = distribution / np.sum(distribution, keepdims=True)

        # threshold at 5% to remove noise
        for i in range(len(EXPERIMENT_DOMAINS)):
            if distribution[i] < 0.05:
                distribution[i] = 0.0

        # normalize again
        distribution = distribution / np.sum(distribution, keepdims=True)

        # now assign to all domains that are at least 34%
        # at most two domains can be above 34%, so this will assign 1 domain, 2 domains or none

        domains = []
        for i in range(len(EXPERIMENT_DOMAINS)):
            if distribution[i] >= 0.34:
                domains.append(EXPERIMENT_DOMAINS[i])

        if len(domains) == 0:
            clusters['none'].add(word)
        else:
            clusters[tuple(domains)].add(word)

    # print clusters
    if PRINT_CLUSTERS:
        first = True
        for cluster, words in clusters.items():
            if not first:
                print()
                print()
            else:
                first = False
            print ('## Cluster', cluster)
            for word in sorted(words):
                print(word)

    if not PRINT_TEMPLATES:
        return

    templates = Counter()
    for sentence, domain, direction in data:
        words = sentence.split()

        if True:
            replaced = []
            for word in words:
                if word.startswith('SLOT_') or word.startswith('SYSTEMACT_') \
                    or word in ('TIME', 'PHONE_NUMBER', 'NUMBER', 'RESERVATION_CODE', 'CONFIRMATION_CODE',
                                'TRAIN_NUMBER', 'POST_CODE'):
                    replaced.append(word)
                    continue

                for clust_key, cluster in clusters.items():
                    if word in cluster:
                        if clust_key == 'none':
                            replaced.append(word)
                        else:
                            replaced.append('WORD_' + '-'.join(clust_key))
                        break
        else:
            replaced = words
        templates[(direction, domain, ' '.join(replaced))] += 1

    for (direction, domain, tmpl), count in templates.most_common():
        print(direction, domain, count, ':', tmpl)


if __name__ == '__main__':
    supervised_cluster()