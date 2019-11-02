#!/usr/bin/env python3

import sys
import json
from collections import defaultdict
import random

random.seed(sys.argv[1])

def parse_belief(belief_str):
    if belief_str == 'none':
        return dict(), []

    tokens = belief_str.split()
    belief = dict()

    domains = set()

    i = 0
    while i < len(tokens):
        domain = tokens[i]
        domains.add(domain)
        i += 1
        slot_name_begin = i
        while tokens[i] != 'is':
            i += 1
        slot_name_end = i
        slot_name = ' '.join(tokens[slot_name_begin : slot_name_end])
        slot_key = domain + '-' + slot_name

        assert tokens[i] == 'is'
        i += 1

        if tokens[i] in ('yes', 'no', 'dontcare'):
            belief[slot_key] = tokens[i]
            i += 1
        else:
            assert tokens[i] == '"'
            i += 1
            slot_value_begin = i
            while tokens[i] != '"':
                i += 1
            slot_value_end = i
            i += 1
            belief[slot_key] = ' '.join(tokens[slot_value_begin : slot_value_end])
    return belief, domains

def belief_to_json(parsed_belief):
    belief_state = []
    for key in parsed_belief:
        belief_state.append({
            'slots': [
                [key, parsed_belief[key]]
            ],
            'act': 'inform'
        })
    return belief_state

dialogues_by_belief_state = defaultdict(list)
dialogues = dict()

shuffle = True

all_lines = list(sys.stdin)
if shuffle:
    random.shuffle(all_lines)

for line in all_lines:
    _id, context, turn, belief_str = line.strip().split('\t')

    system, user = turn.split(';')
    system = system.strip()
    user = user.strip()

    belief_state, belief_domains = parse_belief(belief_str)

    if shuffle:
        turn_obj = {
            'system_transcript': system,
            'transcript': user,
            'belief_state': belief_state,
            'turn_label': None,
            'domain': ''
        }

        if context == 'none':
            turn_obj['turn_idx'] = 0
            dialogue_obj = {
                "dialogue_idx": 'shuffled' + sys.argv[1] + '/' + str(len(dialogues)),
                "domains": list(belief_domains),
                "dialogue": [
                    turn_obj
                ],
            }
            dialogues[dialogue_obj['dialogue_idx']] = dialogue_obj
            dialogues_by_belief_state[belief_str].append(dialogue_obj)
        else:
            candidates = dialogues_by_belief_state[context]
            if len(candidates) == 0:
                continue

            chosen = candidates.pop(random.randint(0, len(candidates)-1))

            turn_obj['turn_idx'] = len(chosen['dialogue'])
            chosen['dialogue'].append(turn_obj)

            dialogues_by_belief_state[belief_str].append(chosen)
    else:
        _, dialogue_idx, turn_idx = _id.split('/')
        turn_idx = int(turn_idx)
        if dialogue_idx not in dialogues:
            dialogues[dialogue_idx] = {
                "dialogue_idx": dialogue_idx,
                "domains": [],
                "dialogue": [],
            }

        dialogue = dialogues[dialogue_idx]

        for domain in belief_domains:
            if domain not in dialogue['domains']:
                dialogue['domains'].append(domain)

        dialogue['dialogue'].append({
            'turn_idx': turn_idx,
            'system_transcript': system,
            'transcript': user,
            'belief_state': belief_state,
            'turn_label': None,
            'domain': ''
        })

for dialogue in dialogues.values():
    dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

    previous_belief_state = dict()
    for turn in dialogue['dialogue']:
        turn_label = dict()
        for key in turn['belief_state']:
            if key not in previous_belief_state or \
                    turn['belief_state'][key] != previous_belief_state[key]:
                turn_label[key] = turn['belief_state'][key]
                domain, slot_name = key.split('-', maxsplit=1)
                turn['domain'] = domain

        turn_label = [[k, v] for k, v in turn_label.items()]
        turn['turn_label'] = turn_label
        previous_belief_state = turn['belief_state']
        turn['belief_state'] = belief_to_json(turn['belief_state'])

print(len(dialogues), file=sys.stderr)
json.dump(list(dialogues.values()), sys.stdout, indent=2)