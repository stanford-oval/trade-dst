
import json
import re
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

def replace_with_system_act(sentence, system_acts):
    sentence = sentence.split()
    for act in system_acts:
        if isinstance(act, str):
            continue

        assert isinstance(act, (list, tuple))
        key, value = act
        if value in ('none', 'yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SYSTEM_' + key))

    return ' '.join(sentence)


def replace_with_slots(sentence, label_dict):
    sentence = sentence.split()
    for key, value in label_dict.items():
        if value in ('none', 'yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SLOT_' + re.sub('[ -]', '_', key)))

    return ' '.join(sentence)


def belief_state_to_string(belief_state):
    slot_str = []
    for slot_key, slot_value in belief_state.items():
        if slot_value == 'none':
            continue
        if slot_key not in ALL_SLOTS:
            continue
        domain, slot_name = slot_key.split('-', maxsplit=1)

        slot_str.append(domain + ' ' + slot_name + ' is')

        if slot_value in ('yes', 'no', 'dontcare'):
            slot_str.append(slot_value)
        else:
            #slot_str.append('" ' + slot_value + ' "')
            slot_str.append('SLOT_' + re.sub('[ -]', '_', slot_key))

    if len(slot_str) == 0:
        return 'none'
    return ' '.join(slot_str)


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


def compute_templates(data):
    templates = Counter()
    distinct = set()

    for dialogue in data:
        dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

        raw_history = []
        history = []
        label_dict = dict()

        for turn in dialogue['dialogue']:
            turn_idx = turn['turn_idx']

            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)

            if turn_idx > 0:
                raw_history.append(turn['system_transcript'])
                system_transcript = replace_with_system_act(turn['system_transcript'], turn['system_acts'])
                history.append(system_transcript)

            raw_history.append(turn['transcript'])
            transcript = replace_with_slots(turn['transcript'], label_dict)
            history.append(transcript)

            if len(label_dict) > 0:
                break

        raw_history = ' <sep> '.join(raw_history)
        history = ' <sep> '.join(history)
        templates[(history, belief_state_to_string(label_dict))] += 1
        distinct.add(raw_history)

    print(len(distinct), file=sys.stderr)
    return templates


def main():
    data = load_data()
    templates = compute_templates(data)

    sum = 0
    for (history, label), count in templates.most_common():
        sum += count
        print(count, history, label, sep='\t')
    print(len(templates), file=sys.stderr)
    print(sum, file=sys.stderr)


if __name__ == '__main__':
    main()