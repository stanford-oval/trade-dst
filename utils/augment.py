"""
Utilities for data augmentation
"""
import sys
from collections import defaultdict
import json
import copy
import re
import random

from .fix_label import fix_general_label_error

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
        fixed_slot_key = slot_key.replace(" ", "").lower() if ("book" not in slot_key) else slot_key.lower()
        slots.append(fixed_slot_key)
        values[fixed_slot_key] = [x.split(' ') for x in ontology_domains[slot_key]]
        assert all(len(x) > 0 for x in values[fixed_slot_key])
    return slots, values
ALL_SLOTS, ALL_VALUES = get_slot_information()


def coin(prob):
    return random.random() < prob


class ReplaceBag:
    def __init__(self):
        self.store = dict()
        self.max_replace_len = 0
        self._used = set()

    def __len__(self):
        return len(self.store)

    def __contains__(self, key):
        return ' '.join(key) in self.store

    def __getitem__(self, key):
        return self.store[' '.join(key)]

    @property
    def used(self):
        return len(self._used)

    def reset_used(self):
        self._used.clear()

    def get_replacement(self, sentence, offset):
        for length in range(1, 1 + min(self.max_replace_len, len(sentence)-offset)):
            slice = ' '.join(sentence[offset:offset+length])
            if slice in self.store:
                self._used.add(slice)
                return length, self.store[slice]
        return 0, None

    def add(self, substring, replacement):
        assert isinstance(substring, list)
        assert len(substring) > 0
        self.store[' '.join(substring)] = replacement
        self.max_replace_len = max(self.max_replace_len, len(substring))


def parse_belief(belief_str):
    if belief_str in ('null', 'greet'):
        return dict(), []

    tokens = belief_str.split()
    belief = dict()

    domains = set()

    i = 0
    # discard intent
    i += 1

    first_domain = tokens[i]
    domains.add(first_domain)
    i += 1

    while i < len(tokens):
        domain = tokens[i]
        domains.add(domain)
        i += 1
        slot_name_begin = i
        while tokens[i] != 'is':
            i += 1
        slot_name_end = i
        slot_name = ' '.join(tokens[slot_name_begin : slot_name_end])

        # this is some weird preprocessing on the slot_name but we do it to be consistent with the TRADE codebase
        slot_name = slot_name.replace(" ", "").lower() if ("book" not in slot_name) else slot_name.lower()

        slot_key = domain + '-' + slot_name

        assert tokens[i] == 'is'
        is_maybe = False
        i += 1

        if tokens[i] == 'maybe':
            is_maybe = True
            i += 1

        if tokens[i] == '?': # ignore questions from the user
            i += 1
            continue

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
            if is_maybe: # ignore maybe values
                continue
            belief[slot_key] = ' '.join(tokens[slot_value_begin : slot_value_end])
            assert belief[slot_key] != '?'
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


def belief_to_dict(json_belief):
    label_dict = dict()
    for slot in json_belief:
        label_dict[slot['slots'][0][0]] = slot['slots'][0][1]
    return label_dict


def compute_signature(label_dict : dict, strict=False):
    sig = []
    for slot_key, slot_value in label_dict.items():
        if slot_value == 'none':
            continue
        if strict or slot_value in ('yes', 'no', 'dontcare'):
            sig.append((slot_key, slot_value))
        else:
            sig.append((slot_key, 'value'))

    sig.sort()
    return '+'.join(slot_key + '=' + slot_value for slot_key, slot_value in sig)


def remove_none_slots(label_dict):
    new_label_dict = dict()
    for slot_key, slot_value in label_dict.items():
        if slot_value == 'none':
            continue
        new_label_dict[slot_key] = slot_value
    return new_label_dict


def compute_prefixes(data):
    prefixes = []

    for dialogue in data:
        domains = set()
        for turn in dialogue['dialogue']:
            turn_idx = turn['turn_idx']

            if 'anything else' in turn['system_transcript']:
               prefixes.append((dialogue, turn_idx, list(domains)))

            domains.add(turn['domain'])

    return prefixes


def compute_continuations(data, strict_signature=False):
    continuations = defaultdict(list)

    for dialogue in data:
        dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

        for turn_idx, turn in enumerate(dialogue['dialogue']):
            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)
            label_dict = remove_none_slots(label_dict)
            if len(label_dict) > 0:
                clone = copy.deepcopy(dialogue)

                clone['dialogue'] = clone['dialogue'][turn_idx+1:]
                continuations[compute_signature(label_dict, strict=strict_signature)].append((clone, label_dict))
                break

    return continuations


def make_replacement_bag(old_label_dict, new_label_dict):
    replace_bag = ReplaceBag()

    for slot_key, old_value in old_label_dict.items():
        if old_value == 'none':
            continue

        new_value = new_label_dict[slot_key]
        if old_value in ('yes', 'no', 'dontcare'):
            assert new_value == old_value
            continue

        replace_bag.add(old_value.split(' '), new_value.split(' '))

    return replace_bag


def apply_replacement(sentence, turn_belief_bag):
    sentence = sentence.split()
    new_sentence = []

    i = 0
    while i < len(sentence):
        replaced_length, replacement = turn_belief_bag.get_replacement(sentence, i)
        if replaced_length > 0:
            new_sentence += replacement
            i += replaced_length
            continue

        new_sentence.append(sentence[i])
        i += 1

    return new_sentence


def apply_new_label_dict_to_dialogue(dialogue, old_label_dict, new_label_dict):
    if old_label_dict == new_label_dict:
        return True

    replace_bag = make_replacement_bag(old_label_dict, new_label_dict)

    for turn in dialogue['dialogue']:
        turn['original_system_transcript'] = turn['system_transcript']
        turn['system_transcript'] = ' '.join(apply_replacement(turn['system_transcript'], replace_bag))
        turn['original_transcript'] = turn['transcript']
        turn['transcript'] = ' '.join(apply_replacement(turn['transcript'], replace_bag))

        label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)

        for slot_key in list(label_dict.keys()):
            if slot_key in old_label_dict and label_dict[slot_key] == old_label_dict[slot_key]:
                if old_label_dict[slot_key] == 'none':
                    del label_dict[slot_key]
                    continue
                label_dict[slot_key] = new_label_dict[slot_key]

        turn['belief_state'] = belief_to_json(label_dict)
        #turn['turn_label'] = [
        #    (slot_key, slot_value) for slot_key, slot_value in label_dict.items()
        #]

    # return whether all values were used in replacement at least once
    return replace_bag.used == len(replace_bag)


def add_continuation(continuations, synth_dialogue, strict_signature=False):
    target_belief = belief_to_dict(synth_dialogue['dialogue'][-1]['belief_state'])
    candidate_continuations = continuations[compute_signature(target_belief, strict=strict_signature)]

    # print(f'Found {len(candidate_continuations)} continuations for {_id}', file=sys.stderr)

    if len(candidate_continuations) > 0:
        chosen_dialogue, old_label_dict = random.choice(candidate_continuations)

        chosen_dialogue = copy.deepcopy(chosen_dialogue)

        chosen_dialogue['dialogue_idx'] = synth_dialogue['dialogue_idx'] + '+' + chosen_dialogue['dialogue_idx']

        if strict_signature:
            assert old_label_dict == target_belief

        if not apply_new_label_dict_to_dialogue(chosen_dialogue, old_label_dict, target_belief):
            return None

        for new_turn_idx, new_turn in enumerate(synth_dialogue['dialogue']):
            chosen_dialogue['dialogue'].insert(new_turn_idx, copy.deepcopy(new_turn))

        return chosen_dialogue
    else:
        if coin(1):
            #print(compute_signature(target_belief, strict=strict_signature), file=sys.stderr)
            #return None
            return copy.deepcopy(synth_dialogue)
        else:
            return None


def add_prefix(prefixes, new_dialogue):
    if len(prefixes) == 0:
        return
    if coin(0.9):
        return

    for _attempt in range(10):
        prefix_dialogue, prefix_turn_idx, prefix_domains = random.choice(prefixes)

        if any(domain in new_dialogue['domains'] for domain in prefix_domains):
            continue

        new_dialogue['dialogue_idx'] += '+prefix:' + prefix_dialogue['dialogue_idx'] + '/' + str(prefix_turn_idx)

        for prev_turn_idx in range(prefix_turn_idx):
            new_dialogue['dialogue'].insert(prev_turn_idx, copy.deepcopy(prefix_dialogue['dialogue'][prev_turn_idx]))

        assert new_dialogue['dialogue'][prefix_turn_idx]['system_transcript'] == ''
        new_dialogue['dialogue'][prefix_turn_idx]['system_transcript'] = prefix_dialogue['dialogue'][prefix_turn_idx]['system_transcript']

        if new_dialogue['dialogue'][prefix_turn_idx]['transcript'].startswith('hello '):
            new_dialogue['dialogue'][prefix_turn_idx]['transcript'] = new_dialogue['dialogue'][prefix_turn_idx]['transcript'][len('hello ! '):]

        if coin(0.2):
            new_dialogue['dialogue'][prefix_turn_idx]['transcript'] = random.choice(['thank you . ', 'thanks ,', 'yes ,']) + new_dialogue['dialogue'][prefix_turn_idx]['transcript']

        if prefix_turn_idx > 0:
            previous_label_dict = fix_general_label_error(prefix_dialogue['dialogue'][prefix_turn_idx-1]['belief_state'], False, ALL_SLOTS)

            for next_turn_idx in range(prefix_turn_idx, len(new_dialogue['dialogue'])):
                turn = new_dialogue['dialogue'][next_turn_idx]
                for slot_key, slot_value in previous_label_dict.items():
                    if any(belief['slots'][0][0] == slot_key for belief in turn['belief_state']):
                        continue
                    turn['belief_state'].append({
                        'slots': [
                            [slot_key, slot_value]
                        ],
                        'act': 'inform'
                    })

        #print(json.dumps(new_dialogue, indent=2), file=sys.stderr)
        #sys.exit(1)
        return new_dialogue


def renumber_turns(new_dialogue):
    previous_belief = dict()
    for turn_idx, turn in enumerate(new_dialogue['dialogue']):
        turn['turn_idx'] = turn_idx

        current_belief = belief_to_dict(turn['belief_state'])
        turn['turn_label'] = [
            (slot_key, slot_value) for slot_key, slot_value in current_belief.items() if
                (slot_key not in previous_belief or previous_belief[slot_key] != slot_value)
        ]

        previous_belief = current_belief


def process_synthetic(prefixes, continuations, from_file, only_domain=None, include_singleton=False,
                      strict_signature=False):
    all_lines = list(from_file)
    random.shuffle(all_lines)

    for line in all_lines:
        _id, sentence, target_code = line.strip().split('\t')
        if target_code == 'none':
            continue

        target_belief, domains = parse_belief(target_code)

        if only_domain is not None and only_domain not in domains:
            continue

        if coin(0.5):
            continue

        synth_dialogue = {
            'dialogue_idx': _id,
            'domains': list(domains),
            'dialogue': [
                {
                    'system_transcript': '',
                    'turn_idx': 0,
                    'belief_state': belief_to_json(target_belief),
                    'transcript': sentence,
                    'system_acts': [],
                    'domain': list(domains)[0]
                }
            ]
        }
        if include_singleton:
            yield synth_dialogue

        new_dialogue = add_continuation(continuations, synth_dialogue, strict_signature=strict_signature)
        if new_dialogue is None:
            continue

        add_prefix(prefixes, new_dialogue)
        renumber_turns(new_dialogue)

        yield new_dialogue


def process_synthetic_json(prefixes, continuations, from_file, only_domain=None):
    synthetic_data = json.load(from_file)
    random.shuffle(synthetic_data)

    for synth_dialogue in synthetic_data:
        if only_domain is not None and only_domain not in synth_dialogue['domains']:
            continue

        if not coin(0.3):
            continue

        new_dialogue = add_continuation(continuations, synth_dialogue)
        if new_dialogue is None:
            continue

        add_prefix(prefixes, new_dialogue)
        renumber_turns(new_dialogue)

        yield new_dialogue


class Augmenter:
    def __init__(self, only_domain=None):
        self.replacements = ReplaceBag()
        self._only_domain = only_domain
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

            # if re.match('[0-9]{,2}:[0-9]{2}$', word, re.IGNORECASE):
            #     new_sentence.append('%02d:%02d' % (random.randint(0, 23), random.randint(0, 59)))
            # elif re.match('\$?[0-9]+$', word, re.IGNORECASE):
            #     if len(word) >= 9:
            #         new_sentence.append('01223%06d' % (random.randint(0, 999999)))
            #     else:
            #         new_sentence.append(str(random.randint(2, 10)))
            # elif re.match('#[0-9a-z]+$', word, re.IGNORECASE):
            #     new_sentence.append('#' + ''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 8)))
            # elif re.match('tr[0-9]+$', word, re.IGNORECASE):
            #     new_sentence.append('tr%04d' % (random.randint(1, 9999)))
            # elif re.match('cb[0-9a-z]{4,}$', word, re.IGNORECASE):
            #     new_sentence.append('cb' + ''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 4)))
            # elif re.match('([a-z]*[0-9]+[a-z]*)+$', word, re.IGNORECASE): # word with number
            #     new_sentence.append(''.join(random.sample('abcdefghijklmnopqrstuvwxyz0123456789', 8)))
            # else:
            #     new_sentence.append(word)
            new_sentence.append(word)
            i += 1

        return new_sentence

    def label_dict_to_replace_bag(self, label_dict : dict):
        replace_bag = ReplaceBag()
        for slot_key, slot_value in label_dict.items():
            if slot_value in ('yes', 'no', 'none', 'dontcare', "do n't care"):
                continue

            domain, _ = slot_key.split('-', maxsplit=1)
            if self._only_domain is not None and self._only_domain != domain:
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
            if 'original_system_transcript' not in turn:
                turn['original_system_transcript'] = system_transcript
            turn['system_transcript'] = ' '.join(self.augment_system(system_transcript, turn_belief_bag))

            user_transcript = turn['transcript']
            if 'original_transcript' not in turn:
                turn['original_transcript'] = user_transcript
            turn['transcript'] = ' '.join(self.augment_user(user_transcript, turn_belief_bag))

            label_dict.update(self.new_slot_values)
            turn['belief_state'] = belief_to_json(label_dict)
            turn['turn_label'] = [
                (slot_key, slot_value) for slot_key, slot_value in label_dict.items()
            ]