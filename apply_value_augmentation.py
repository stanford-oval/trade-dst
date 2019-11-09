#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import re
import copy
import random

random.seed(1235)

from utils.fix_label import fix_general_label_error

def coin(prob):
    return random.random() < prob

EXPERIMENT_DOMAINS = ["none", "hotel", "train", "restaurant", "attraction", "taxi"]
DOMAIN_INDICES = dict()
for domain in EXPERIMENT_DOMAINS:
    DOMAIN_INDICES[domain] = len(DOMAIN_INDICES)
def get_slot_information():
    ontology = json.load(open("data/clean-ontology.json", 'r'))
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])

    slots = []
    values = dict()
    for slot_key in ontology_domains:
        fixed_slot_key = slot_key.replace(" ","").lower() if ("book" not in slot_key) else slot_key.lower()
        slots.append(fixed_slot_key)
        values[fixed_slot_key] = [x.split(' ') for x in ontology_domains[slot_key]]
        assert all(len(x) > 0 for x in values[fixed_slot_key])
    return slots, values
ALL_SLOTS, ALL_VALUES = get_slot_information()

def label_dict_to_belief(label_dict):
    belief_state = []
    for key, value in label_dict.items():
        belief_state.append({
            'slots': [
                [key, value]
            ],
            'act': 'inform'
        })
    return belief_state


class ReplaceBag:
    def __init__(self):
        self.store = dict()
        self.max_replace_len = 0

    def __len__(self):
        return len(self.store)

    def __contains__(self, key):
        return ' '.join(key) in self.store

    def __getitem__(self, key):
        return self.store[' '.join(key)]

    def get_replacement(self, sentence, offset):
        for length in range(1, 1 + min(self.max_replace_len, len(sentence)-offset)):
            slice = ' '.join(sentence[offset:offset+length])
            if slice in self.store:
                return length, self.store[slice]
        return 0, None

    def add(self, substring, replacement):
        assert isinstance(substring, list)
        assert len(substring) > 0
        self.store[' '.join(substring)] = replacement
        self.max_replace_len = max(self.max_replace_len, len(substring))


class Augmenter:
    def __init__(self):
        self.replacements = ReplaceBag()
        self.new_slot_values = dict()

    def augment_user(self, sentence, turn_belief_bag : ReplaceBag):
        sentence = sentence.split()
        new_sentence = []

        i = 0
        while i < len(sentence):
            # if we already replaced this span, apply the same replacement to be consistent
            replaced_length, replacement = self.replacements.get_replacement(sentence, i)
            if replaced_length > 0:
                #print('stored replacement', sentence[i : i + replaced_length], '->', replacement)
                new_sentence += replacement
                i += replaced_length
                continue

            # replace punctuation
            if sentence[i] in ('.', '?', ',', '!'):
                if coin(0.05):
                    new_sentence.append(random.choice(('.', '?', ',', '!')))
                    i += 1
                    continue
                elif coin(0.05):
                    i += 1
                    continue

            # if this is a slot value, replace it with other slot values
            replaced_length, slot_name = turn_belief_bag.get_replacement(sentence, i)
            if replaced_length > 0 and coin(0.8):
                replacement = random.choice(ALL_VALUES[slot_name])
                assert isinstance(replacement, list)
                assert len(replacement) > 0
                # remember for the future
                self.replacements.add(sentence[i : i + replaced_length], replacement)
                # update the annotation on the dialogue
                self.new_slot_values[slot_name] = ' '.join(replacement)

                #print('new replacement', sentence[i : i + replaced_length], '->', replacement)
                new_sentence += replacement
                i += replaced_length
                continue

            new_sentence.append(sentence[i])
            i += 1

        return new_sentence

    def augment_system(self, sentence, turn_belief_bag : ReplaceBag):
        sentence = sentence.split()
        new_sentence = []

        i = 0
        while i < len(sentence):
            # if this is a slot value, do not replace it:
            #
            # either the user previously introduced it, and this is the new value (by chance)
            # or the user is about to introduce it picking it up from the system sentence (rare but happens)
            # in that case, we don't want to make changes
            replaced_length, slot_name = turn_belief_bag.get_replacement(sentence, i)
            if replaced_length > 0:
                new_sentence += sentence[i: i + replaced_length]
                i += replaced_length
                continue

            # if we already replaced this span, apply the same replacement to be consistent
            replaced_length, replacement = self.replacements.get_replacement(sentence, i)
            if replaced_length > 0:
                #print('stored replacement', sentence[i : i + replaced_length], '->', replacement)
                new_sentence += replacement
                i += replaced_length
                continue

            # replace punctuation
            if sentence[i] in ('.', '?', ',', '!'):
                if coin(0.05):
                    new_sentence.append(random.choice(('.', '?', ',', '!')))
                    i += 1
                    continue
                elif coin(0.05):
                    i += 1
                    continue

            # replace other value words with random words
            word = sentence[i]

            if re.match('[0-9]{,2}:[0-9]{2}$', word, re.IGNORECASE):
                new_sentence.append('%02d:%02d' % (random.randint(0, 23), random.randint(0, 59)))
            elif re.match('\$?[0-9]+$', word, re.IGNORECASE):
                if len(word) >= 9:
                    new_sentence.append('01223%06d' % (random.randint(0, 999999)))
                else:
                    new_sentence.append(str(random.randint(2, 10)))
            elif re.match('#[0-9a-z]+$', word, re.IGNORECASE):
                new_sentence.append('#' + ''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 8)))
            elif re.match('tr[0-9]+$', word, re.IGNORECASE):
                new_sentence.append('tr%04d' % (random.randint(1, 9999)))
            elif re.match('cb[0-9a-z]{4,}$', word, re.IGNORECASE):
                new_sentence.append('cb' + ''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 4)))
            elif re.match('([a-z]*[0-9]+[a-z]*)+$', word, re.IGNORECASE): # word with number
                new_sentence.append(''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 8)))
            else:
                new_sentence.append(word)
            i += 1

        return new_sentence

    def label_dict_to_replace_bag(self, label_dict : dict):
        replace_bag = ReplaceBag()
        for slot_key, slot_value in label_dict.items():
            if slot_value in ('yes', 'no', 'none', 'dontcare', "do n't care"):
                continue

            slot_value = slot_value.split(' ')
            if slot_value in self.replacements:
                label_dict[slot_key] = ' '.join(self.replacements[slot_value])
                continue

            replace_bag.add(slot_value, slot_key)
        return replace_bag

    def augment(self, dialogue):
        for turn in dialogue:
            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)
            label_dict.update(self.new_slot_values)

            turn_belief_bag = self.label_dict_to_replace_bag(label_dict)

            system_transcript = turn['system_transcript']
            turn['original_system_transcript'] = system_transcript
            turn['system_transcript'] = ' '.join(self.augment_system(system_transcript, turn_belief_bag))

            user_transcript = turn['transcript']
            turn['original_transcript'] = user_transcript
            turn['transcript'] = ' '.join(self.augment_user(user_transcript, turn_belief_bag))

            label_dict.update(self.new_slot_values)
            turn['belief_state'] = label_dict_to_belief(label_dict)
            del turn['turn_label']

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