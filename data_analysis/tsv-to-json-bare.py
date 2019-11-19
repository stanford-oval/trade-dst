#!/usr/bin/env python3

import sys
import json
import random
from collections import defaultdict

from utils.augment import coin, parse_belief, belief_to_json, compute_signature, process_synthetic

random.seed(12345)

def compute_continuations(from_file):
    all_lines = list(from_file)
    random.shuffle(all_lines)

    continuations = defaultdict(list)

    for line in all_lines:
        _id, context_code, sentence, target_code = line.strip().split('\t')
        if target_code == 'none' or context_code == 'none':
            continue

        #if not coin(0.1):
        #    continue

        context_belief, _ = parse_belief(context_code)
        target_belief, domains = parse_belief(target_code)

        system, user = sentence.split(' <sep> ')

        new_turn = {
            'system_transcript': system,
            'turn_idx': 0,
            'belief_state': belief_to_json(target_belief),
            'transcript': user,
            'system_acts': [],
            'domain': list(domains)[0]
        }
        new_dialogue = {
            'dialogue_idx': _id,
            'dialogue': [new_turn],
            'domains': list(domains)
        }

        continuations[compute_signature(context_belief, strict=True)].append((new_dialogue, context_belief))

    return continuations

def main():
    with open(sys.argv[2]) as fp:
        continuations = compute_continuations(fp)

    new_data = []

    for _ in range(2):
        with open(sys.argv[1]) as fp:
            for new_dialogue in process_synthetic([], continuations, from_file=fp, include_singleton=True,
                                                  strict_signature=True):
                new_data.append(new_dialogue)

    json.dump(new_data, sys.stdout, indent=2)
    print()
    print(len(new_data), file=sys.stderr)


if __name__ == '__main__':
    main()