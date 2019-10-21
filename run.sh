#!/usr/bin/env bash

./k8s/train.sh --experiment multiwoz --dataset baseline --model bert-baseline -- -dec=TRADE -bsz=16 -dr=0.2 -lr=0.001 -le=1 --encoder BERT --bert_model bert-base-uncased