
import json
import sys

from utils.augment import ALL_SLOTS
from utils.fix_label import fix_general_label_error

def main():
    data = json.load(sys.stdin)

    new_data = []
    for dialogue in data:
        all_domains = set(dialogue['domains'])
        # add sometimes missing domains to annotation
        for turn in dialogue['dialogue']:
            turn_belief_dict = fix_general_label_error(turn["belief_state"], False, ALL_SLOTS)
            for slot_key, slot_value in turn_belief_dict:
                if slot_value == 'none':
                    continue
                domain, slot_name = slot_key.split('-', maxsplit=1)
                all_domains.add(domain)

        if sys.argv[1] in all_domains:
            continue
        new_data.append(dialogue)

    print(len(data), len(new_data), file=sys.stderr)
    json.dump(new_data, sys.stdout, indent=2)

if __name__ == '__main__':
    main()