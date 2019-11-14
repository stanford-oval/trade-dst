#!/usr/bin/env python3
import copy
import sys
import json
import random

from utils.fix_label import fix_general_label_error
from utils.augment import EXPERIMENT_DOMAINS, ALL_SLOTS, ReplaceBag, compute_continuations, process_synthetic, \
    apply_replacement, belief_to_json, remove_none_slots

random.seed(12345)

TRANSFER_PHRASES = {
    'taxi': [
        'taxi',
        'taxi ride',
        'cab'
    ],
    'train': [
        'train',
        'train ride',
        'train reservation'
    ],

    # FIXME
    'restaurant': [
        'restaurant',
        'restaurant reservation',
        'food place',
        'place to eat'
    ],
    'hotel': [
        'hotel',
        'place to stay'
    ],
    'attraction': []
}

def load_data(except_domain):
    filename = 'data/train_dials.json'

    filtered_domains = []
    with open(filename) as fp:
        data = json.load(fp)

        for dialogue in data:
            is_good_domain = True
            for domain in dialogue['domains']:
                if domain not in EXPERIMENT_DOMAINS \
                    or domain == except_domain:
                    is_good_domain = False
            if is_good_domain:
                filtered_domains.append(dialogue)
    return filtered_domains


def transfer_data(original_data, from_domain, to_domain):
    new_data = []

    transfer_phrases_from = TRANSFER_PHRASES[from_domain]
    transfer_phrases_to = TRANSFER_PHRASES[to_domain]
    assert len(transfer_phrases_from) > 0
    assert len(transfer_phrases_to) > 0

    transfer_replace_bag = ReplaceBag()
    for phrase in transfer_phrases_from:
        transfer_replace_bag.add(phrase.split(' '), random.choice(transfer_phrases_to).split(' '))

    for dialogue in original_data:
        if not from_domain in dialogue['domains']:
            new_data.append(dialogue)
            continue

        original_dialogue = copy.deepcopy(dialogue)
        new_data.append(original_dialogue)

        good_dialogue = True
        for turn in dialogue['dialogue']:
            turn_idx = int(turn['turn_idx'])

            if turn_idx > 0:
                turn['original_system_transcript'] = turn['system_transcript']
                turn['system_transcript'] = ' '.join(apply_replacement(turn['system_transcript'], transfer_replace_bag))
                turn['original_transcript'] = turn['transcript']
                turn['transcript'] = ' '.join(apply_replacement(turn['transcript'], transfer_replace_bag))
            found_transfer_phrase = transfer_replace_bag.used > 0

            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)
            label_dict = remove_none_slots(label_dict)

            found_transfer_slot = False
            found_bad_slot = False
            new_label_dict = dict()
            for slot_key, slot_value in label_dict.items():
                domain, slot_name = slot_key.split('-', maxsplit=1)

                # we have removed all "to_domain" data so if we see this it's a mislabel, ignore as bad
                if domain == to_domain:
                    good_dialogue = False
                    break
                if domain != from_domain:
                    new_label_dict[slot_key] = slot_value
                    continue

                found_transfer_slot = True
                new_slot_key = to_domain + '-' + slot_name
                if new_slot_key not in ALL_SLOTS:
                    found_bad_slot = True
                    break
                new_label_dict[new_slot_key] = slot_value

            turn['belief_state'] = belief_to_json(new_label_dict)
            turn['turn_label'] = [
                (slot_key, slot_value) for slot_key, slot_value in new_label_dict.items()
            ]
            turn['domain'] = to_domain if turn['domain'] == from_domain else turn['domain']

            if found_bad_slot or (found_transfer_slot and not found_transfer_phrase):
                good_dialogue = False
                break

        if good_dialogue:
            dialogue['dialogue_idx'] = dialogue['dialogue_idx'] + '/' + from_domain + '->' + to_domain
            dialogue['domains'] = [x for x in dialogue['domains'] if x != from_domain] + [to_domain]

            new_data.append(dialogue)

            #print(json.dumps(dialogue['dialogue'], indent=2))
            #sys.exit(0)

    return new_data

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <from-domain> <to-domain>")
        sys.exit(1)

    from_domain = sys.argv[1]
    to_domain = sys.argv[2]

    original_data = load_data(to_domain)
    original_data = transfer_data(original_data, from_domain, to_domain)

    continuations = compute_continuations(original_data)

    new_data = []
    new_data += original_data

    for new_dialogue in process_synthetic(continuations, from_file=sys.stdin):
        new_data.append(new_dialogue)

    json.dump(new_data, sys.stdout, indent=2)
    print()
    print(len(new_data), file=sys.stderr)


if __name__ == '__main__':
    main()