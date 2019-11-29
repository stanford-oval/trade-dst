#!/usr/bin/env python3

import json
import sys
import copy
import random
import collections
import numpy as np
from sklearn import cluster

from utils.eval_utils import get_joint_accuracy, get_name_accuracy, print_turn
from utils.augment import EXPERIMENT_DOMAINS, ALL_SLOTS

ErrorTuple = collections.namedtuple('ErrorTuple', ('dialogue', 'turn_idx', 'annotation', 'prediction', 'features'))


random.seed(1235)


def uses_slot(slot_name, prediction):
    slot_name = slot_name + '-'
    for slot in prediction:
        if slot.startswith(slot_name):
            return True
    return False


def feature_uses_slot_in_annotation_or_prediction(slot_name):
    return lambda error: (uses_slot(slot_name, error.annotation) or uses_slot(slot_name, error.prediction))


def feature_has_domain(domain):
    return lambda error: domain in error.dialogue['domains']


ALL_FEATURES = collections.OrderedDict()

def feature(fn):
    ALL_FEATURES[fn.__name__] = fn
    return fn


@feature
def number_of_slots_in_annotation(error):
    return len(error.annotation)


@feature
def turn_number(error):
    return error.turn_idx


@feature
def correct_slot_names(error):
    return get_name_accuracy(error.annotation, error.prediction)


@feature
def annotation_is_subset(error):
    for slot in error.annotation:
        if not slot in error.prediction:
            return False
    return True


@feature
def prediction_is_subset(error):
    for slot in error.prediction:
        if not slot in error.annotation:
            return False
    return True


@feature
def error_has_dontcare(error):
    incorrect_slots = set()
    for slot in error.prediction:
        if slot in error.annotation:
            continue
        incorrect_slots.add(slot)
    for slot in error.annotation:
        if slot in error.prediction:
            continue
        incorrect_slots.add(slot)
    return any(slot.endswith('-dontcare') for slot in incorrect_slots)


@feature
def error_has_boolean(error):
    incorrect_slots = set()
    for slot in error.prediction:
        if slot in error.annotation:
            continue
        incorrect_slots.add(slot)
    for slot in error.annotation:
        if slot in error.prediction:
            continue
        incorrect_slots.add(slot)
    return any(slot.endswith('-yes') or slot.endswith('-no') for slot in incorrect_slots)


def load_features():
    for slot_key in ALL_SLOTS:
        ALL_FEATURES['uses_slot_in_annotation_or_prediction:' + slot_key] = feature_uses_slot_in_annotation_or_prediction(slot_key)
    for domain in EXPERIMENT_DOMAINS:
        ALL_FEATURES['has_domain:' + domain] = feature_has_domain(domain)
load_features()


def load_data():
    filename = 'data/dev_dials.json'

    dialogue_dev_data = dict()
    with open(filename) as fp:
        for dialogue in json.load(fp):
            dialogue['dialogue'].sort(key=lambda x: int(x['turn_idx']))
            dialogue_dev_data[dialogue['dialogue_idx']] = dialogue

    return dialogue_dev_data


def load_predictions(all_models):
    pred_data_all = dict()
    for modelname in all_models:
        with open('./model-' + modelname + '/predictions/full/prediction_TRADE_dev.json') as fp:
            pred_data = json.load(fp)
            pred_data_all[modelname] = pred_data
    return pred_data_all


def compute_errors(dialogue_dev_data, pred_data):
    errors = []

    for dialogue_id, dialogue in dialogue_dev_data.items():
        for turn in dialogue['dialogue']:
            turn_idx = turn['turn_idx']
            turn_idx_str = str(turn_idx)
            annotation = pred_data[dialogue_id][turn_idx_str]['turn_belief']
            annotation.sort()
            prediction = pred_data[dialogue_id][turn_idx_str]['pred_bs_ptr']
            prediction.sort()

            if not get_joint_accuracy(pred_data[dialogue_id][turn_idx_str]):
                errors.append(ErrorTuple(dialogue, turn_idx, annotation, prediction, features=dict()))

    return errors


def featurize_errors(errors):
    X = np.zeros((len(errors), len(ALL_FEATURES)), dtype=np.float32)

    for error_idx, error in enumerate(errors):
        for feature_idx, (feature_name, feature_func) in enumerate(ALL_FEATURES.items()):
            error.features[feature_name] = feature_func(error)
            X[error_idx, feature_idx] = float(error.features[feature_name])

    #for error in errors[:10]:
    #    print(error.dialogue['dialogue_idx'], '/', error.turn_idx)
    #    for feature in ALL_FEATURES:
    #        print(feature, '=', error.features[feature])
    #    print()

    return X


def main():
    modelname = 'baseline21'

    dialogue_dev_data = load_data()
    pred_data_all = load_predictions([modelname])
    pred_data = pred_data_all[modelname]
    print('loaded data', file=sys.stderr)

    errors = compute_errors(dialogue_dev_data, pred_data)
    random.shuffle(errors)
    print('computed all errors', file=sys.stderr)

    error_features = featurize_errors(errors)
    print('featurized all errors', file=sys.stderr)

    clustering = cluster.AffinityPropagation(preference=-1000, max_iter=1000, verbose=True)
    clustering.fit(error_features)
    N_CLUSTERS = len(clustering.cluster_centers_)

    #N_CLUSTERS = 8
    #clustering = cluster.SpectralClustering(N_CLUSTERS, random_state=42)
    #clustering.fit(error_features)
    print('clustered', file=sys.stderr)

    #for error_idx, error in enumerate(errors):
    #    print(error.dialogue['dialogue_idx'], error.turn_idx, clustering.labels_[error_idx], sep='\t')

    cluster_sizes = [0] * N_CLUSTERS
    clusters = [[] for _ in range(N_CLUSTERS)]
    for error_idx in range(len(errors)):
        cluster_idx = clustering.labels_[error_idx]
        cluster_sizes[cluster_idx] += 1
        clusters[cluster_idx].append(errors[error_idx])

    for cluster_idx in range(N_CLUSTERS):

        if cluster_idx > 0:
            print()
            print()
        print(f'# Cluster #{cluster_idx}: {cluster_sizes[cluster_idx]} elements')

        center = clustering.cluster_centers_[cluster_idx]
        for feature_idx, feature in enumerate(ALL_FEATURES):
            if center[feature_idx] != 0.0:
                print(feature, '=', center[feature_idx])

        for error in clusters[cluster_idx][:5]:
            print_turn(dialogue_dev_data, pred_data, error.dialogue['dialogue_idx'], error.turn_idx)

        break

if __name__ == '__main__':
    main()