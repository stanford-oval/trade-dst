#!/usr/bin/env python3

import json
import sys
import copy
import random
import collections
from functools import partial
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.svm
import sklearn.linear_model

from utils.eval_utils import get_joint_accuracy, get_name_accuracy, print_turn, remove_none_slots
from utils.augment import EXPERIMENT_DOMAINS, ALL_SLOTS

ExampleTuple = collections.namedtuple('ExampleTuple',
                                      ('dialogue', 'turn_idx', 'annotation', 'prediction', 'previous',
                                       'label', 'features'))
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


def substring_seq(string, substring):
    for i in range(len(string) - len(substring) + 1):
        good = True
        for j in range(len(substring)):
            if string[i + j] != substring[j]:
                good = False
                break
        if good:
            return i
    return -1


ALL_FEATURES = collections.OrderedDict()

def feature(fn):
    ALL_FEATURES[fn.__name__] = fn
    return fn


def per_domain_feature(fn):
    for domain in EXPERIMENT_DOMAINS:
        if domain == 'none':
            continue
        ALL_FEATURES[fn.__name__ + ':' + domain] = partial(fn, domain)
    return fn


def per_slot_feature(fn):
    for slot_key in ALL_SLOTS:
        ALL_FEATURES[fn.__name__ + ':' + slot_key] = partial(fn, slot_key)
    return fn


def range_feature(max_num):
    def decorator(fn):
        for num in range(max_num):
            ALL_FEATURES[fn.__name__ + ':' + str(num)] = partial(fn, num, exact=True)
        ALL_FEATURES[fn.__name__ + ':>=' + str(max_num)] = partial(fn, max_num, exact=False)
    return decorator


#@per_slot_feature
def uses_slot_in_annotation_or_prediction(slot_name, example):
    return uses_slot(slot_name, example.annotation) or uses_slot(slot_name, example.prediction)


@per_slot_feature
def uses_slot_in_annotation(slot_name, example):
    return uses_slot(slot_name, example.annotation)


#@per_slot_feature
def slot_is_wrong(slot_name, example):
    return get_value(slot_name, example.annotation) != get_value(slot_name, example.prediction)


@per_domain_feature
def has_domain(domain, example):
    return domain in example.dialogue['domains']


@per_slot_feature
def slot_value_appears_in_dialogue(slot_key, example):
    slot_value = get_value(slot_key, example.annotation)
    if slot_value in ('yes', 'no', 'dontcare', 'none'):
        return False

    slot_value = slot_value.split()
    for turn in example.dialogue['dialogue']:
        turn_idx = turn['turn_idx']
        if turn_idx > example.turn_idx:
            break
        if substring_seq(turn['system_transcript'].split(), slot_value) != -1:
            return True
        if substring_seq(turn['transcript'].split(), slot_value) != -1:
            return True
    return False


@per_slot_feature
def slot_value_appears_in_user(slot_key, example):
    slot_value = get_value(slot_key, example.annotation)
    if slot_value in ('yes', 'no', 'dontcare', 'none'):
        return False

    slot_value = slot_value.split()
    for turn in example.dialogue['dialogue']:
        turn_idx = turn['turn_idx']
        if turn_idx > example.turn_idx:
            break
        if substring_seq(turn['transcript'].split(), slot_value) != -1:
            return True
    return False


@per_slot_feature
def annotation_type_error(slot_key, example):
    slot_value = get_value(slot_key, example.annotation)
    if slot_value in ('dontcare', 'none'):
        return False

    if slot_key in ('hotel-parking', 'hotel-internet'):
        return slot_value not in ('yes', 'no')

    return slot_value in ('yes', 'no')


@range_feature(10)
def number_of_slots_in_annotation(num, example, exact=True):
    if exact:
        return len(example.annotation) == num
    else:
        return len(example.annotation) >= num


@range_feature(5)
def turn_number(num, example, exact=True):
    if exact:
        return example.turn_idx == num
    else:
        return example.turn_idx >= num


#@feature
def correct_slot_names(example):
    return get_name_accuracy(example.annotation, example.prediction)


#@feature
def annotation_is_subset(example):
    for slot in example.annotation:
        if not slot in example.prediction:
            return False
    return True


#@feature
def prediction_is_subset(example):
    for slot in example.prediction:
        if not slot in example.annotation:
            return False
    return True


@feature
def previous_turn_correct(example):
    if example.previous is None:
        return True
    else:
        return example.previous.label > 0.0


@feature
def has_dontcare(example):
    #return any(slot.endswith('-dontcare') for slot in example.prediction) or \
    return any(slot.endswith('-dontcare') for slot in example.annotation)


@feature
def has_boolean(example):
    #return any(slot.endswith('-yes') or slot.endswith('-no') for slot in example.prediction) or \
    return any(slot.endswith('-yes') or slot.endswith('-no') for slot in example.annotation)


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

    previous = None
    for dialogue_id, dialogue in dialogue_dev_data.items():
        for turn in dialogue['dialogue']:
            turn_idx = turn['turn_idx']
            turn_idx_str = str(turn_idx)
            annotation = list(remove_none_slots(pred_data[dialogue_id][turn_idx_str]['turn_belief']))
            annotation.sort()
            prediction = list(remove_none_slots(pred_data[dialogue_id][turn_idx_str]['pred_bs_ptr']))
            prediction.sort()

            joint_accuracy = get_joint_accuracy(pred_data[dialogue_id][turn_idx_str])

            example = ExampleTuple(dialogue, turn_idx, annotation, prediction,
                                   previous=previous, label=joint_accuracy, features=dict())
            examples.append(example)
            previous = example

    return examples


def featurize_examples(examples):
    X = np.zeros((len(examples), 2*len(ALL_FEATURES)), dtype=np.float32)
    Y = np.empty((len(examples),), dtype=np.float32)

    for example_idx, example in enumerate(examples):
        for feature_idx, (feature_name, feature_func) in enumerate(ALL_FEATURES.items()):
            example.features[feature_name] = feature_func(example)
            X[example_idx, feature_idx] = float(example.features[feature_name])
            example.features['!' + feature_name] = 1 - X[example_idx, feature_idx]
            X[example_idx, len(ALL_FEATURES) + feature_idx] = 1 - X[example_idx, feature_idx]
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

    all_features = list(ALL_FEATURES) + ['!' + feature for feature in ALL_FEATURES]
    for feature_idx, feature in enumerate(all_features):
        print(f'{feature} = {classifier.coef_[0, feature_idx]}')


def do_heuristic(dialogue_dev_data, pred_data, examples, example_features, example_labels):
    all_features = list(ALL_FEATURES) + ['!' + feature for feature in ALL_FEATURES]
    feature_counts = np.sum(example_features, axis=0)
    feature_error_count = np.zeros((len(all_features),), dtype=np.float32)

    for example_idx, example in enumerate(examples):
        if example_labels[example_idx] < 1.0:
            feature_error_count += example_features[example_idx]

    feature_error_percent = feature_error_count / (1e-5 + feature_counts)

    feature_error_percent = [(feature_idx, feature_error_percent[feature_idx], feature_counts[feature_idx])
                             for feature_idx in range(len(all_features))]
    feature_error_percent.sort(key=lambda x: (x[1], x[2]))

    for feature_idx, error_pct, feature_cnt in feature_error_percent:
        print('%s = %.0f / %.0f = %.2f' % (all_features[feature_idx],
                                           feature_error_count[feature_idx],
                                           feature_cnt,
                                           100 * error_pct))


def do_greedy(dialogue_dev_data, pred_data, examples):
    all_features = list(ALL_FEATURES) + ['!' + feature for feature in ALL_FEATURES]

    examples_total = len(examples)
    examples_left = examples_total
    cluster_idx = 0
    while examples_left / examples_total > 0.05:
        feature_counts = np.zeros((len(all_features),), dtype=np.float32)
        feature_error_count = np.zeros((len(all_features),), dtype=np.float32)

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
    modelname = 'baseline21'
    #modelname = 'aug5'

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