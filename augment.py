#!/usr/bin/env python3

import json
import sys
import copy
import random

from utils.augment import Augmenter, EXPERIMENT_DOMAINS, compute_prefixes, compute_continuations, process_synthetic_json

random.seed(1235)

def load_data():
    filename = 'data/train_dials.json'

    filtered_domains = []
    with open(filename) as fp:
        data = json.load(fp)

        for dialogue in data:
            dialogue['dialogue'].sort(key=lambda x: int(x['turn_idx']))

            is_good_domain = True
            for domain in dialogue['domains']:
                if domain not in EXPERIMENT_DOMAINS:
                    is_good_domain = False
            if is_good_domain:
                filtered_domains.append(dialogue)
    return filtered_domains


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <synthetic.json>")
        sys.exit(1)

    synthetic_json = sys.argv[1]

    original_data = load_data()
    prefixes = compute_prefixes(original_data)
    continuations = compute_continuations(original_data)

    new_data = []
    new_data += original_data

    with open(synthetic_json) as fp:
        for new_dialogue in process_synthetic_json(prefixes, continuations, from_file=fp):
            new_data.append(new_dialogue)

    json.dump(new_data, sys.stdout, indent=2)
    print()
    print(len(new_data), file=sys.stderr)


if __name__ == '__main__':
    main()