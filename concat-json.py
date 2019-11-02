#!/usr/bin/env python3

import json
import sys

flatten = []
for fn in sys.argv[1:]:
    flatten.extend(json.load(open(fn)))

json.dump(flatten, sys.stdout)