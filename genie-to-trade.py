#!/usr/bin/python3

import json
import sys

from utils.augment import parse_belief, belief_to_json


def main():
    with open(sys.argv[1]) as fp:
        genie_data = json.load(fp)

    output = []
    for dialogue in genie_data:
        domains = set()

        new_dialogue = {
            'dialogue_idx': 'synth' + dialogue['id'],
            'dialogue': [
            ]
        }

        previous_belief = dict()
        for turn_idx, turn in enumerate(dialogue['turns']):
            belief, turn_domains = parse_belief(turn['target'])

            new_dialogue['dialogue'].append({
                'system_transcript': turn['system'],
                'turn_idx': turn_idx,
                'belief_state': belief_to_json(belief),
                'turn_label': [
                    (slot_key, slot_value) for slot_key, slot_value in belief.items() if
                        (slot_key not in previous_belief or previous_belief[slot_key] != slot_value)
                ],
                'transcript': turn['user'],
                'system_acts': [],
                'domain': list(turn_domains)[0] if len(turn_domains) else ''
            })

            previous_belief = belief
            domains.update(turn_domains)

        new_dialogue['domains'] = list(domains)
        assert len(new_dialogue['domains']) > 0

        for turn in new_dialogue['dialogue']:
            if turn['domain'] == '':
                turn['domain'] = new_dialogue['domains'][0]

        output.append(new_dialogue)

    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == '__main__':
    main()