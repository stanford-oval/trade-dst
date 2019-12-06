#!/usr/bin/python3

import json
import sys
from collections import defaultdict

IGNORED_KEYS = {
    'price', 'choice', 'leave', 'arrive', 'parking', 'internet', 'name',
    'day', 'depart', 'type', 'food', 'area' , 'dest'
}

def main():
    data = json.load(sys.stdin)

    ontology = defaultdict(set)

    for dialogue in data:
        for turn in dialogue['dialogue']:
            for sysact in turn['system_acts']:
                if not isinstance(sysact, (list, tuple)):
                    continue

                key, value = sysact
                if key == 'none' or key in IGNORED_KEYS:
                    continue
                ontology[key].add(value)

    ontology2 = dict()
    for key in ontology:
        ontology2[key] = list(ontology[key])

    json.dump(ontology2, fp=sys.stdout, indent=2)

    print(list(ontology.keys()), file=sys.stderr)


if __name__ == '__main__':
    main()