#!/usr/bin/env python3

import json
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import re
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(BASE_DIR, '..')
os.chdir(repo_root)

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


def preprocess(sentence):
    sentence = sentence.split()

    for word in sentence:
        if word in ('.', '?'):
            continue

        if re.match('[0-9]{,2}:[0-9]{2}', word, re.IGNORECASE):
            yield 'TIME'
        elif re.match('\$?[0-9]+', word, re.IGNORECASE):
            if len(word) >= 9:
                yield 'PHONE_NUMBER'
            else:
                yield 'NUMBER'
        elif re.match('#[0-9a-z]+', word, re.IGNORECASE):
            yield 'RESERVATION_CODE'
        elif re.match('tr[0-9]+', word, re.IGNORECASE):
            yield 'TRAIN_NUMBER'
        elif re.match('cb[0-9a-z]{4,}', word, re.IGNORECASE):
            yield 'POST_CODE'
        elif re.match('([a-z]*[0-9]+[a-z]*)+', word, re.IGNORECASE): # word with number
            yield 'CONFIRMATION_NUMBER'
        else:
            yield word


def replace_with_system_act(sentence, system_acts):
    sentence = sentence.split()
    for act in system_acts:
        if isinstance(act, str):
            continue

        assert isinstance(act, (list, tuple))
        key, value = act
        if value in ('yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SYSTEM_' + key))

    return ' '.join(sentence)


def replace_with_slots(sentence, turn_label, label_dict):
    sentence = sentence.split()
    for slot in turn_label:
        assert isinstance(slot, (list, tuple))
        key, _ = slot
        value = label_dict[key]
        if value in ('yes', 'no', 'dontcare'):
            continue

        sentence = list(replace_with(sentence, value, 'SLOT_' + re.sub('[ -]', '_', key)))

    return ' '.join(sentence)

def difference(new, old):
    new_set = set(new)
    old_set = set(old)
    added = list(set(new_set)-set(old_set))
    return added

def load_data():
    filename = 'data/train_dials.json'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    with open(filename) as fp:
        data = json.load(fp)

    for dialogue in data:
        dialogue_idx = dialogue['dialogue_idx']

        dialogue['dialogue'].sort(key=lambda x: x['turn_idx'])

        prev_turn = 'none'

        for turn in dialogue['dialogue']:
            system_transcript = ' '.join(preprocess(turn['system_transcript']))
            transcript = ' '.join(preprocess(turn['transcript']))

            label_dict = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)

            if prev_turn in EXPERIMENT_DOMAINS and system_transcript.strip() != '':
                yield replace_with_system_act(system_transcript, turn['system_acts']), prev_turn, 'system'

            if turn['domain'] in EXPERIMENT_DOMAINS:
                yield replace_with_slots(transcript, turn['turn_label'], label_dict), turn['domain'], 'user'

            prev_turn = turn['domain']

def get_full_belief(turn):
    dict_of_slots = fix_general_label_error(turn['belief_state'], False, ALL_SLOTS)
    return ['-'.join((el, dict_of_slots[el])) for el in dict_of_slots], list(dict_of_slots.keys())
    
def generate_turn_frame(data):    
    
    slot_updates = dict()
    
    for dialogue in data:
        

        d_idx = dialogue["dialogue_idx"].split('.')[0]
        
        for t_idx, turn in enumerate(dialogue["dialogue"]):
            
            idx = '_'.join((d_idx, str(t_idx)))
            slot_updates[idx] = dict()
            belief, slots = get_full_belief(turn)
            added_belief = ['-'.join(el) for el in turn["turn_label"]]
            added_slots = [el[0] for el in turn["turn_label"]]
            
            slot_updates[idx] = {'dialogue': d_idx, 'turn': t_idx, 'full_belief': belief, \
                                  'step_belief': added_belief, 'full_slots': slots, 'step_slots': added_slots, \
                                  'transcript': turn['transcript'], 'system_transcript': turn['system_transcript'], \
                                  'step_empty': (len(added_belief) == 0), 'full_empty': (len(belief) == 0), \
                                  'domain': turn['domain']}
            
    slot_updates = pd.read_json(json.dumps(slot_updates)).transpose()
    return slot_updates

    
def select_domains(frame, domain_list):
    
    return frame[frame.domain.isin(domain_list)]

def dials_as_frame(split_type, domains = None):
    
    assert(split_type in ["train", "test", "dev"])

    filename = os.path.join(repo_root, 'data', ''.join((split_type, '_dials.json')))

    with open(filename) as fp:
        dialogue_data = json.load(fp)
    
    frame = generate_turn_frame(dialogue_data)
    
    if domains:
        frame = select_domains(frame, domains)

    return frame

def get_errors(df):
    df_correct = df["det_full_correct"].apply(sum).astype(float)
    df_slots = df["det_full_correct"].apply(len).astype(float)
    df["percent_correct"] = df_correct/df_slots
    a = df[["turn", "percent_correct"]]
    partially_correct = a[(a["percent_correct"]>0) & (a["percent_correct"] < 1)]
    fully_correct = a[a["percent_correct"] == 1]
    fully_incorrect = a[a["percent_correct"] == 0]
    correct_empty = a[a["percent_correct"].isna()]
    return partially_correct, fully_correct, fully_incorrect, correct_empty

'''
TODO: This checks for full <slot type - value> accuracy, doesn't split it...
'''
def add_error_types(frame):
    
    pred_step_belief = frame['pred_step_belief']
    true_step_belief = frame['true_step_belief']
    
    pred_full_belief = frame['pred_full_belief']
    true_full_belief = frame['true_full_belief']


    frame["det_inserted"] = list(set(pred_step_belief) - set(true_step_belief))
    frame["det_missed"] = list(set(true_step_belief) - set(pred_step_belief))
    frame["det_full_correct"] = [el in pred_full_belief for el in true_full_belief]
    frame["det_step_correct"] = [el in pred_step_belief for el in true_step_belief]
    
    det_step_correct = frame["det_step_correct"]
    det_full_correct = frame["det_full_correct"]
    det_inserted = frame["det_inserted"]
    det_missed = frame["det_missed"]
    
    frame["step_correct"] = (False not in det_step_correct)#bool(sum(det_step_correct))
    frame["full_correct"] = (False not in det_full_correct) #bool(sum(det_full_correct))  
    frame["inserted"] = (len(det_inserted)>0)
    frame["missed"] = (len(det_missed)>0)
    if len(det_full_correct) > 0:
        frame["percent_found"] = sum(det_full_correct)/len(det_full_correct)
    else:
        frame["percent_found"] = None
    
    return frame

def generate_TRADE_turn_frame(predictions, pred_slot_columns=False, gt_slot_columns=False):       

    slot_updates = dict()
    for d_idx, dialogue in predictions.items():
        old_belief = []
        true_old_belief = []
        for t_idx in range(len(dialogue.keys())):
            # the unique id: <dialogue>_<turn>
            idx = '_'.join((d_idx.split('.')[0], str(t_idx+1)))
            t_idx = str(t_idx)
            slot_updates[idx] = dict()
            turn = dialogue[t_idx]
            new_belief = turn['pred_bs_ptr']
            true_belief = turn['turn_belief']
            added_belief = difference(new_belief, old_belief)
            true_added_belief = difference(true_belief, true_old_belief)
            
            # has anything been added?
            added_empty = (len(added_belief) == 0)
            true_empty = (len(true_added_belief) == 0)
            
            slot_updates[idx]['dialogue'] = d_idx.split('.')[0]
            slot_updates[idx]['turn'] = t_idx
            
            slot_updates[idx]['pred_full_belief'] = new_belief
            slot_updates[idx]['true_full_belief'] = true_belief
            slot_updates[idx]['pred_step_belief'] = added_belief 
            slot_updates[idx]['true_step_belief'] = true_added_belief
            
            slot_updates[idx] = add_error_types(slot_updates[idx])
            
            slot_updates[idx]['pred_empty'] = added_empty
            slot_updates[idx]['true_empty'] = true_empty
            
            old_belief = new_belief
            true_old_belief = true_belief
            
    slot_updates = pd.read_json(json.dumps(slot_updates, indent=4)).transpose()
    return slot_updates

def experiment_results_frame(input_file):
    output_file = os.path.join(experiment_path(experiment), "inference_turn_info.csv")
    baseline_test_set = read_json(input_file)
    frame = generate_TRADE_turn_frame(baseline_test_set)
    return frame
