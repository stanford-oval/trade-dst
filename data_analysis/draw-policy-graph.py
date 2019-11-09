#!/usr/bin/env python3

import json
import sys
from collections import Counter

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

def fix_none_typo(value):
    if value in ("not men", "not", "not mentioned", "", "not mendtioned", "fun", "art"):
        return 'none'
    else:
        return value

def get_node_key_slot_names(label_dict):
    slots = []
    domains = set()

    for slot_key, slot_value in label_dict.items():
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        domains.add(slot_key.split('-')[0])
        slots.append(slot_key.replace(' ', '-'))

    if len(slots) == 0:
        return 'none', 'none'

    return ','.join(domains), ','.join(slots)
def get_node_key_slot_names_delta(turn_label):
    slots = []
    domains = set()

    for slot_key, slot_value in turn_label:
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        domains.add(slot_key.split('-')[0])
        slots.append(slot_key.replace(' ', '-'))

    if len(slots) == 0:
        return 'none', 'none'

    return ','.join(domains), ','.join(slots)
def get_node_key_slot_counts(label_dict):
    slots = Counter()

    for slot_key, slot_value in label_dict.items():
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        slot_domain = slot_key.split('-')[0]
        slots[slot_domain] += 1

    if len(slots) == 0:
        return 'none'
    return ','.join(domain + '=' + str(count) for domain, count in slots.items())
def get_node_key_slot_domains(label_dict):
    slots = set()

    for slot_key, slot_value in label_dict.items():
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        slot_domain = slot_key.split('-')[0]
        slots.add(slot_domain)

    if len(slots) == 0:
        return 'none'
    return ','.join(slots)
def get_node_key_slot_counts_delta(turn_label):
    slots = Counter()

    for slot_key, slot_value in turn_label:
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        slot_domain = slot_key.split('-')[0]
        slots[slot_domain] += 1

    if len(slots) == 0:
        return 'none'

    return ','.join(domain + '=' + str(count) for domain, count in slots.items())
def get_node_key_domains_delta(turn_label):
    slots = set()

    for slot_key, slot_value in turn_label:
        slot_value = fix_none_typo(slot_value)
        if slot_value == 'none':
            continue
        slot_domain = slot_key.split('-')[0]
        slots.add(slot_domain)

    if len(slots) == 0:
        return 'none'

    return ','.join(slots)

def load_data():
    filename = 'data/train_dials.json'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    with open(filename) as fp:
        data = json.load(fp)

    nodes = Counter()
    edges = Counter()

    for dialogue in data:
        prev_node_key = 'none'
        for turn in dialogue['dialogue']:
            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)

            #domains, node_key = get_node_key_slot_names(label_dict)
            domains, node_key = get_node_key_slot_names_delta(turn['turn_label'])
            if domains not in ('none', 'restaurant'):
                continue
            #node_key = get_node_key_domains_delta(turn['turn_label'])
            #node_key = get_node_key_slot_domains(label_dict)
            nodes[node_key] += 1
            edges[(prev_node_key, node_key)] += 1

            if prev_node_key == 'train' and node_key == 'attraction':
                print(json.dumps(dialogue, indent=2), file=sys.stderr)

            prev_node_key = node_key

    return nodes, edges

def main():
    nodes, edges = load_data()

    print('strict digraph states {')

    node_num = dict()
    for i, (node, node_count) in enumerate(nodes.most_common()):
        #if ',' in node:
        #    continue
        node_num[node] = i

        node_label = node.replace('restaurant-', '')

        #node_count = nodes[node]
        print(f's{i} [label="{node_label}"]; # {node_count}')

    for (_from, _to), count in edges.items():
        #if ',' in _from or ',' in _to:
        #    continue

        print(f's{node_num[_from]} -> s{node_num[_to]} [label="{count}"];')

    print('}')

if __name__ == '__main__':
    main()