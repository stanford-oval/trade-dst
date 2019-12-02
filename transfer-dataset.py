#!/usr/bin/env python3
import copy
import sys
import json
import random

from utils.fix_label import fix_general_label_error
from utils.augment import EXPERIMENT_DOMAINS, ALL_SLOTS, ReplaceBag, compute_prefixes, compute_continuations, process_synthetic_json, \
    apply_replacement, belief_to_json, remove_none_slots, Augmenter

random.seed(12345)

TRANSFER_PHRASES = {
    'taxi': [
        'taxi ride',
        'taxi',
        'cab',
        'car'
    ],
    'train': [
        'train ride',
        'train reservation',
        'train ticket',
        'train',
    ],

    # FIXME
    'restaurant': [
        'restaurant reservation',
        'restaurant',
        'food place',
        'place to eat'
    ],
    'hotel': [
        'hotel',
        'place to stay'
    ],
    'attraction': []
}

def load_data(except_domain, keep_pct=0):
    filename = 'data/train_dials.json'

    filtered_domains = []
    in_domain_data = []

    with open(filename) as fp:
        data = json.load(fp)

        for dialogue in data:
            dialogue['dialogue'].sort(key=lambda x: int(x['turn_idx']))

            all_domains = set(dialogue['domains'])
            # add sometimes missing domains to annotation
            for turn in dialogue['dialogue']:
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, ALL_SLOTS)
                for slot_key, slot_value in turn_belief_dict.items():
                    if slot_value == 'none':
                        continue
                    domain, slot_name = slot_key.split('-', maxsplit=1)
                    all_domains.add(domain)
            dialogue['domains'] = list(all_domains)
            dialogue['domains'].sort()

            is_good_domain = True
            is_in_domain = False
            for domain in dialogue['domains']:
                if domain not in EXPERIMENT_DOMAINS:
                    is_good_domain = False
                if domain == except_domain:
                    is_in_domain = True
            if not is_good_domain:
                continue

            if is_in_domain:
                in_domain_data.append(dialogue)
            else:
                filtered_domains.append(dialogue)

    if keep_pct > 0:
        random.shuffle(in_domain_data)
        to_keep = int(keep_pct * len(in_domain_data))
        return filtered_domains + in_domain_data[:to_keep]
    return filtered_domains


def transfer_data(original_data, from_domain, to_domain):
    new_data = []

    transfer_phrases_from = TRANSFER_PHRASES[from_domain]
    transfer_phrases_to = TRANSFER_PHRASES[to_domain]
    assert len(transfer_phrases_from) > 0
    assert len(transfer_phrases_to) > 0


    for dialogue in original_data:
        new_data.append(dialogue)
        if not from_domain in dialogue['domains']:
            continue

        new_dialogue = copy.deepcopy(dialogue)

        transfer_replace_bag = ReplaceBag()
        for phrase in transfer_phrases_from:
            transfer_replace_bag.add(phrase.split(' '), random.choice(transfer_phrases_to).split(' '))

        good_dialogue = True
        for turn in new_dialogue['dialogue']:
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
            new_dialogue['dialogue_idx'] = new_dialogue['dialogue_idx'] + '/' + from_domain + '->' + to_domain
            new_dialogue['domains'] = [x for x in new_dialogue['domains'] if x != from_domain] + [to_domain]

            # replace the values in the new dialogue with values that make sense for the domain
            augmenter = Augmenter(only_domain=to_domain)
            augmenter.augment(new_dialogue['dialogue'])

            new_data.append(new_dialogue)

            #print(json.dumps(dialogue['dialogue'], indent=2))
            #sys.exit(0)

    return new_data


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <synthetic.json> <from-domain> <to-domain> [<keep-pct>] [<do-transfer>]")
        sys.exit(1)

    synthetic_json = sys.argv[1]
    from_domain = sys.argv[2]
    to_domain = sys.argv[3]
    if len(sys.argv) > 4:
        keep_pct = float(sys.argv[4])
        if keep_pct > 0 and keep_pct < 1:
            print('Argument keep_pct should be between 0 and 100', file=sys.stderr)
            sys.exit(1)
        keep_pct /= 100
    else:
        keep_pct = 0
    if len(sys.argv) > 5:
        do_transfer = sys.argv[5] in ('yes', 'True', '1')
    else:
        do_transfer = True

    original_data = load_data(to_domain, keep_pct)

    prefixes = compute_prefixes(original_data)

    if do_transfer:
        original_data = transfer_data(original_data, from_domain, to_domain)

    continuations = compute_continuations(original_data)

    new_data = []
    new_data += original_data

    with open(synthetic_json) as fp:
        for new_dialogue in process_synthetic_json(prefixes, continuations, from_file=fp, only_domain=to_domain):
            new_data.append(new_dialogue)

    json.dump(new_data, sys.stdout, indent=2)
    print()
    print(len(new_data), file=sys.stderr)


if __name__ == '__main__':
    main()