#!/usr/bin/env python3

import json
import sys
import copy
import random

random.seed(1235)

from utils.augment import Augmenter, EXPERIMENT_DOMAINS


def load_data():
    filename = 'data/train_dials.json'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    filtered_domains = []
    with open(filename) as fp:
        data = json.load(fp)

        for dialogue in data:
            is_good_domain = True
            for domain in dialogue['domains']:
                if domain not in EXPERIMENT_DOMAINS:
                    is_good_domain = False
            if is_good_domain:
                filtered_domains.append(dialogue)
    return filtered_domains


def apply_augmentation(data, augidx):
    data = copy.deepcopy(data)

    for dialogue in data:
        dialogue['dialogue_idx'] = dialogue['dialogue_idx'] + '/' + str(augidx)

        dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])
        augmenter = Augmenter()
        augmenter.augment(dialogue['dialogue'])

    return data


def main():
    data = load_data()

    everything = []
    for i in range(15):
        everything += data

    for i in range(5):
        everything += apply_augmentation(data, i)

    print(json.dumps(everything, indent=2))

if __name__ == '__main__':
    main()