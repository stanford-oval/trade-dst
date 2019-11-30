#!/usr/bin/env python3

import json
import sys
import copy
import random
import collections
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.svm
import sklearn.linear_model

from utils.eval_utils import get_joint_accuracy, get_name_accuracy, print_turn, remove_none_slots
from utils.augment import EXPERIMENT_DOMAINS, ALL_SLOTS

ExampleTuple = collections.namedtuple('ExampleTuple',
                                      ('dialogue', 'turn_idx', 'annotation', 'prediction', 'label', 'features'))
np.random.seed(1234)
random.seed(1236)


def uses_slot(slot_name, prediction):
    slot_name = slot_name + '-'
    for slot in prediction:
        if slot.startswith(slot_name):
            return True
    return False


def get_value(slot_key, prediction):
    for slot in prediction:
        domain, slot_name, slot_value = slot.split('-')
        if domain + '-' + slot_name == slot_key:
            return slot_value
    return 'none'


def feature_uses_slot_in_annotation_or_prediction(slot_name):
    return lambda example: (uses_slot(slot_name, example.annotation) or uses_slot(slot_name, example.prediction))


def feature_uses_slot_in_annotation(slot_name):
    return lambda example: (uses_slot(slot_name, example.annotation))


def feature_slot_is_wrong(slot_name):
    return lambda example: get_value(slot_name, example.annotation) != get_value(slot_name, example.prediction)


def feature_has_domain(domain):
    return lambda example: domain in example.dialogue['domains']


ALL_FEATURES = collections.OrderedDict()

def feature(fn):
    ALL_FEATURES[fn.__name__] = fn
    return fn


def feature_number_of_slots_in_annotation(num):
    return lambda example: len(example.annotation) == num


def feature_turn_number(num):
    return lambda example: example.turn_idx == num


#@feature
def correct_slot_names(example):
    return get_name_accuracy(example.annotation, example.prediction)


#@feature
def annotation_is_subset(error):
    for slot in error.annotation:
        if not slot in error.prediction:
            return False
    return True


#@feature
def prediction_is_subset(error):
    for slot in error.prediction:
        if not slot in error.annotation:
            return False
    return True


@feature
def has_dontcare(example):
    #return any(slot.endswith('-dontcare') for slot in example.prediction) or \
    return any(slot.endswith('-dontcare') for slot in example.annotation)


@feature
def has_boolean(example):
    #return any(slot.endswith('-yes') or slot.endswith('-no') for slot in example.prediction) or \
    return any(slot.endswith('-yes') or slot.endswith('-no') for slot in example.annotation)


def make_turn_number_features(max_turn_number):
    for turn_idx in range(max_turn_number):
        ALL_FEATURES['turn_number:' + str(turn_idx)] = feature_turn_number(turn_idx)
    ALL_FEATURES['turn_number:>=' + str(max_turn_number)] = lambda example: example.turn_idx >= max_turn_number


def make_number_of_slots_in_annotation_feature(max_num_slots):
    for num_slots in range(max_num_slots):
        ALL_FEATURES['number_of_slots_in_annotation:' + str(num_slots)] = \
            feature_number_of_slots_in_annotation(num_slots)
    ALL_FEATURES['number_of_slots_in_annotation:>=' + str(max_num_slots)] = \
        lambda example: len(example.annotation) >= max_num_slots


def load_features():
    for slot_key in ALL_SLOTS:
        #ALL_FEATURES['uses_slot_in_annotation_or_prediction:' + slot_key] = feature_uses_slot_in_annotation_or_prediction(slot_key)
        #ALL_FEATURES['slot_is_wrong:' + slot_key] = feature_slot_is_wrong(slot_key)
        ALL_FEATURES['uses_slot_in_annotation:' + slot_key] = feature_uses_slot_in_annotation(slot_key)
    for domain in EXPERIMENT_DOMAINS:
        if domain == 'none':
            continue
        ALL_FEATURES['has_domain:' + domain] = feature_has_domain(domain)
    make_turn_number_features(5)
    make_number_of_slots_in_annotation_feature(10)
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


def compute_examples(dialogue_dev_data, pred_data):
    examples = []

    for dialogue_id, dialogue in dialogue_dev_data.items():
        for turn in dialogue['dialogue']:
            turn_idx = turn['turn_idx']
            turn_idx_str = str(turn_idx)
            annotation = list(remove_none_slots(pred_data[dialogue_id][turn_idx_str]['turn_belief']))
            annotation.sort()
            prediction = list(remove_none_slots(pred_data[dialogue_id][turn_idx_str]['pred_bs_ptr']))
            prediction.sort()

            joint_accuracy = get_joint_accuracy(pred_data[dialogue_id][turn_idx_str])
            examples.append(ExampleTuple(dialogue, turn_idx, annotation, prediction,
                                         label=joint_accuracy, features=dict()))

    return examples


def featurize_examples(examples):
    X = np.zeros((len(examples), len(ALL_FEATURES)), dtype=np.float32)
    Y = np.empty((len(examples),), dtype=np.float32)

    for example_idx, example in enumerate(examples):
        for feature_idx, (feature_name, feature_func) in enumerate(ALL_FEATURES.items()):
            example.features[feature_name] = feature_func(example)
            X[example_idx, feature_idx] = float(example.features[feature_name])
            Y[example_idx] = float(example.label)

    #for error in errors[:10]:
    #    print(error.dialogue['dialogue_idx'], '/', error.turn_idx)
    #    for feature in ALL_FEATURES:
    #        print(feature, '=', error.features[feature])
    #    print()

    return X, Y


def do_cluster(dialogue_dev_data, pred_data, errors, error_features):
    clustering = sklearn.cluster.AffinityPropagation(max_iter=10000, verbose=True)
    clustering.fit(error_features)
    N_CLUSTERS = len(clustering.cluster_centers_)

    # N_CLUSTERS = 8
    # clustering = cluster.SpectralClustering(N_CLUSTERS, random_state=42)
    # clustering.fit(error_features)
    print('clustered', file=sys.stderr)

    # for error_idx, error in enumerate(errors):
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


def do_lda(dialogue_dev_data, pred_data, errors, error_features):
    N_CLUSTERS = 5
    lda = sklearn.decomposition.LatentDirichletAllocation(n_components=N_CLUSTERS,
                                                          random_state=1234,
                                                          max_iter=100,
                                                          evaluate_every=5,
                                                          verbose=1)

    transformed_errors = lda.fit_transform(error_features)

    cluster_index = np.argmax(transformed_errors, axis=1)

    cluster_sizes = [0] * N_CLUSTERS
    clusters = [[] for _ in range(N_CLUSTERS)]
    for error_idx in range(len(errors)):
        cluster_idx = cluster_index[error_idx]
        cluster_sizes[cluster_idx] += 1
        clusters[cluster_idx].append(errors[error_idx])

    topic_assign_prob = lda.components_ / np.sum(lda.components_, axis=0, keepdims=True)

    for cluster_idx in range(N_CLUSTERS):
        print()
        print()
        print(f'# Cluster #{cluster_idx}: {cluster_sizes[cluster_idx]} elements')

        for feature_idx, feature in enumerate(ALL_FEATURES):
            print(feature, '=', topic_assign_prob[cluster_idx, feature_idx])

        #for error in clusters[cluster_idx][:5]:
        #    print_turn(dialogue_dev_data, pred_data, error.dialogue['dialogue_idx'], error.turn_idx)


def do_classifier(dialogue_dev_data, pred_data, examples, example_features, example_labels):
    classifier = sklearn.linear_model.LogisticRegression(class_weight='balanced', verbose=1)

    classifier.fit(example_features, example_labels)

    print()
    print(classifier.score(example_features, example_labels))

    for feature_idx, feature in enumerate(ALL_FEATURES):
        print(f'{feature} = {classifier.coef_[0, feature_idx]}')


def do_heuristic(dialogue_dev_data, pred_data, examples, example_features, example_labels):
    feature_counts = np.sum(example_features, axis=0)
    feature_error_count = np.zeros((len(ALL_FEATURES),), dtype=np.float32)

    for example_idx, example in enumerate(examples):
        if example_labels[example_idx] < 1.0:
            feature_error_count += example_features[example_idx]

    feature_error_percent = feature_error_count / feature_counts

    feature_error_percent = [(feature_idx, feature_error_percent[feature_idx], feature_counts[feature_idx])
                             for feature_idx in range(len(ALL_FEATURES))]
    feature_error_percent.sort(key=lambda x: (x[1], x[2]))

    all_features = list(ALL_FEATURES)

    for feature_idx, error_pct, feature_cnt in feature_error_percent:
        print('%s = %.0f / %.0f = %.2f' % (all_features[feature_idx],
                                           feature_error_count[feature_idx],
                                           feature_cnt,
                                           100 * error_pct))


def do_greedy(dialogue_dev_data, pred_data, examples):
    all_features = list(ALL_FEATURES)

    examples_total = len(examples)
    examples_left = examples_total
    cluster_idx = 0
    while examples_left / examples_total > 0.1:
        feature_counts = np.zeros((len(ALL_FEATURES),), dtype=np.float32)
        feature_error_count = np.zeros((len(ALL_FEATURES),), dtype=np.float32)

        for example_idx, example in enumerate(examples):
            for feature_idx, feature in enumerate(all_features):
                feature_value = float(example.features[feature])
                feature_counts[feature_idx] += feature_value
                if example.label < 1.0:
                    feature_error_count[feature_idx] += feature_value

        feature_error_percent = feature_error_count / (1e-5 + feature_counts)

        worst_feature_idx = np.argmax(feature_error_percent)
        worst_feature = all_features[worst_feature_idx]

        print('Cluster %d: %s = %.0f / %.0f / %.0f = %.2f' % (
            cluster_idx, worst_feature, feature_error_count[worst_feature_idx],
            feature_counts[worst_feature_idx], examples_left,
            feature_error_percent[worst_feature_idx]
        ))

        examples = [ex for ex in examples if float(ex.features[worst_feature]) == 0.0]
        examples_left = len(examples)

        cluster_idx += 1


def do_count_features(example_features):
    feature_counts = np.sum(example_features != 0, axis=0)
    for feature_idx, feature in enumerate(ALL_FEATURES):
        print(f'{feature} = {feature_counts[feature_idx]}')


def main():
    #modelname = 'baseline21'
    modelname = 'aug5'

    dialogue_dev_data = load_data()
    pred_data_all = load_predictions([modelname])
    pred_data = pred_data_all[modelname]
    print('loaded data', file=sys.stderr)

    examples = compute_examples(dialogue_dev_data, pred_data)
    random.shuffle(examples)
    print('computed all examples', file=sys.stderr)

    errors = [ex for ex in examples if ex.label != 1.0]
    print('num examples =', len(examples))
    print('num errors =', len(errors))

    example_features, example_labels = featurize_examples(examples)
    print('featurized all examples', file=sys.stderr)

    #do_count_features(example_features)
    #do_cluster(dialogue_dev_data, pred_data, errors, example_features)
    #do_lda(dialogue_dev_data, pred_data, examples, example_features)
    #do_classifier(dialogue_dev_data, pred_data, examples, example_features, example_labels)
    #do_heuristic(dialogue_dev_data, pred_data, examples, example_features, example_labels)
    do_greedy(dialogue_dev_data, pred_data, examples)

if __name__ == '__main__':
    main()