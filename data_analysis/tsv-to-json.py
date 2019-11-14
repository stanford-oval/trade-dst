#!/usr/bin/env python3

import sys
import json
import random

from utils.augment import EXPERIMENT_DOMAINS, compute_continuations, process_synthetic

random.seed(12345)


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

def main():
    original_data = load_data()
    continuations = compute_continuations(original_data)

    new_data = []
    for i in range(3):
        new_data += original_data

    for new_dialogue in process_synthetic(continuations, from_file=sys.stdin):
        new_data.append(new_dialogue)

    json.dump(new_data, sys.stdout, indent=2)
    print()
    print(len(new_data), file=sys.stderr)


if __name__ == '__main__':
    main()