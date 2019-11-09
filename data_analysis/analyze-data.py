#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
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
        if word in ('.', '?'):
            continue

        if re.match('[0-9]{,2}:[0-9]{2}', word, re.IGNORECASE):
            yield 'TIME'
        elif re.match('\$?[0-9]+', word, re.IGNORECASE):
            if len(word) >= 9:
                yield 'PHONE_NUMBER'
            else:
                yield 'NUMBER'
        elif re.match('#[0-9a-z]+', word, re.IGNORECASE):
            yield 'RESERVATION_CODE'
        elif re.match('tr[0-9]+', word, re.IGNORECASE):
            yield 'TRAIN_NUMBER'
        elif re.match('cb[0-9a-z]{4,}', word, re.IGNORECASE):
            yield 'POST_CODE'
        elif re.match('([a-z]*[0-9]+[a-z]*)+', word, re.IGNORECASE): # word with number
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

def difference(new, old):
    new_set = set(new)
    old_set = set(old)
    added = list(set(new_set)-set(old_set))
    return added

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

def generate_turn_frame(data):    
    
    slot_updates = dict()
    
    for dialogue in data:
        
        old_belief = []
        old_slots = []
        d_idx = dialogue["dialogue_idx"].split('.')[0]
        
        for t_idx, turn in enumerate(dialogue["dialogue"]):
            
            idx = '_'.join((d_idx, str(t_idx)))
            slot_updates[idx] = dict()
            belief = ['-'.join(el) for el in dialogue["dialogue"][t_idx]["turn_label"]]
            slots = [el[0] for el in dialogue["dialogue"][t_idx]["turn_label"]]
            added_belief = difference(belief, old_belief)
            added_slots = difference(slots, old_slots)
            
            slot_updates[idx] = {'dialogue': d_idx, 'turn': t_idx, 'full_belief': belief, \
                                  'step_belief': added_belief, 'full_slots': slots, 'step_slots': added_slots, \
                                  'transcript': turn['transcript'], 'system_transcript': turn['system_transcript'], \
                                  'step_empty': (len(added_belief) == 0), 'full_empty': (len(belief) == 0)}
            
            old_belief = belief
            old_slots = slots
            
    slot_updates = pd.read_json(json.dumps(slot_updates)).transpose()
    return slot_updates

# def main():
#     for