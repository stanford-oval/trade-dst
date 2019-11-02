#!/usr/bin/env python3

import sys
import json

from utils.fix_label import fix_general_label_error

data = json.load(sys.stdin)

ontology = json.load(open("data/multi-woz/MULTIWOZ2.1/ontology.json", 'r'))
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS
ALL_SLOTS = get_slot_information(ontology)

def belief_state_to_string(belief_state, history):
    fix_general_label_error(belief_state, False, ALL_SLOTS)

    slot_str = []
    for slot in belief_state:
        assert len(slot['slots']) == 1
        assert slot['act'] == 'inform'

        slot_key = slot['slots'][0][0]
        slot_value = slot['slots'][0][1]
        if slot_value == 'none':
            continue

        if slot_key not in ALL_SLOTS:
            continue
        domain, slot_name = slot_key.split('-', maxsplit=1)

        slot_str.append(domain + ' ' + slot_name + ' is')

        if slot_value in ('yes', 'no', 'dontcare'):
            slot_str.append(slot_value)
        else:
            slot_str.append('" ' + slot_value + ' "')

    return ' '.join(slot_str)

for dialogue in data:
    dialogue_idx = dialogue['dialogue_idx']

    # Filtering and counting domains
    filter_domain = False
    for domain in dialogue["domains"]:
        if domain not in EXPERIMENT_DOMAINS:
            filter_domain = True
            break
    if filter_domain:
        continue

    dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

    context = ''
    history = ''

    for turn in dialogue['dialogue']:
        # genie interprets letters at the beginning of the ID as flags
        # prepend a / to avoid that
        turn_id = '/' + dialogue_idx + '/' + str(turn['turn_idx'])

        turn_sentences = turn['system_transcript'] + ' ; ' + turn['transcript']
        history += ' ' + turn_sentences

        turn_belief = belief_state_to_string(turn['belief_state'], history)

        print(turn_id, context, turn_sentences, turn_belief, sep='\t')
        context = turn_belief