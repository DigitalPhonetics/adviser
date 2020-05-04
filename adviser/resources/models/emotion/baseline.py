###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

"""This script trains an SVM on either acoustic, visual or combined features.

Run python baseline.py -h to see commandline arguments and usage.
"""
import argparse
import copy
import itertools
from joblib import dump
from joblib import load
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from termcolor import cprint

import data_utils


p = argparse.ArgumentParser()
p.add_argument('modality', choices=['visual', 'audio', 'audiovisual'],
               type=str)
p.add_argument('--audio_data', type=str, help='path to audio features')
p.add_argument('--visual_data', type=str, help='path to visual features')
# SVM parameters:
p.add_argument('-c', '--penalty', help='Penalty parameter C for SVM',
               type=float, default=1.0)
p.add_argument('-d', '--degree', type=int, default=1,
               help='Polynomial degree of SVM with polynomial kernel, \
               ignored for other kernels')
p.add_argument('-k', '--kernel', type=str, default='rbf',
               choices=['linear', 'poly', 'rbf'])
# MLP parameters:
p.add_argument('--hidden_layers', type=str, default='100')
p.add_argument('--alpha', type=float, default=0.0001)
# general command line arguments:
p.add_argument('--load_model', type=str, help='path to stored model')
p.add_argument('--label_type', choices=['category', 'arousal', 'valence'],
               type=str, default='category')
p.add_argument('--loso', action='store_true',
               help='turn on leave-one-speaker-out cross validation for MSP-IMPROV')
ARGS = p.parse_args()
if ARGS.modality == 'visual' and ARGS.visual_data is None:
    p.error('visual modality requires --visual_data')
if ARGS.modality == 'audio' and ARGS.audio_data is None:
    p.error('audio modality requires --audio_data')
if ARGS.modality == 'audiovisual' and (ARGS.visual_data is None or
                                       ARGS.audio_data is None):
    p.error('audiovisual modality requires --visual_data and --audio_data')
cprint('Parameters: {}'.format(ARGS), 'yellow')

# SVM
C = ARGS.penalty
KERNEL = ARGS.kernel
DEGREE = ARGS.degree
# MLP
HIDDEN_LAYERS = tuple([int(x) for x in ARGS.hidden_layers.split(',')])
ALPHA = ARGS.alpha

MODALITY = ARGS.modality
# 4 classes in MSP-IMPROV:
LABELS = {'happiness': 1, 'neutral': 2, 'sadness': 3, 'anger': 0}
# 4+1 classes in IEMOCAP:
# LABELS = {'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3, 'ang': 0}
# LABELS = {'hap': 5, 'exc': 2, 'neu': 6, 'sad': 8, 'ang': 0, 'fru': 4}
# LABELS = {'hap': 5, 'exc': 2, 'neu': 6, 'sad': 8, 'ang': 0}
# CREMA-D classes:
# LABELS = {'A': 0, 'H': 1, 'N': 2, 'S': 3}
AUDIO_DATA = ARGS.audio_data
VISUAL_DATA = ARGS.visual_data
LABEL_TYPE = ARGS.label_type
LOSO = ARGS.loso


def conf_matrix_percentages(conf):
    """Convert a confusion matrix with absolute count into percentages."""
    total = np.sum(conf, axis=1).reshape(conf.shape[0], 1)
    percentages = conf / total.astype('float32') * 100
    return np.around(percentages, decimals=2)


def group_continuous_labels(cont_labels):
    """Convert continuous labels into classes."""
    new_lbls = copy.deepcopy(cont_labels)
    new_lbls[new_lbls <= 2] = 0.0  # original threshold 2
    new_lbls[(new_lbls > 2) & (new_lbls < 4)] = 1.0
    new_lbls[new_lbls >= 4] = 2.0  # original threshold 4
    return new_lbls


def get_data_and_labels(dataset, label_type, label_map):
    if label_type == 'category':
        idx = np.isin(dataset.get_category(), list(label_map.keys()))
        data = dataset.get_features()[idx]
        class_lbls = dataset.get_category()[idx]
        return data, class_lbls
    elif label_type == 'arousal':
        class_lbls = group_continuous_labels(dataset.get_arousal())
    elif label_type == 'valence':
        class_lbls = group_continuous_labels(dataset.get_valence())
    # if label_type is arousal or valence, take all feature data
    data = dataset.get_features()
    return data, class_lbls


def score(clf, train_x, train_y, dev_x, dev_y, test_x, test_y):
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)
    test_pred = clf.predict(test_x)

    train_score = clf.score(train_x, train_y)
    train_uar = sklearn.metrics.balanced_accuracy_score(train_y, train_pred)
    dev_score = clf.score(dev_x, dev_y)
    dev_uar = sklearn.metrics.balanced_accuracy_score(dev_y, dev_pred)
    test_score = clf.score(test_x, test_y)
    test_uar = sklearn.metrics.balanced_accuracy_score(test_y, test_pred)
    print(f'\nTraining accuracy: {train_score}, UAR: {train_uar}')
    print(f'Dev accuracy: {dev_score}, UAR: {dev_uar}')
    print(f'Test accuracy: {test_score}, UAR: {test_uar}')

    train_confusion = sklearn.metrics.confusion_matrix(train_y, train_pred)
    print('Train set confusion matrix:\n', train_confusion)

    confusion = sklearn.metrics.confusion_matrix(test_y, test_pred)
    print('\nConfusion Matrix on Test set\n(labels in sorted order by default')
    print('columns are predictions, rows gold standard\n', confusion)
    confusion_percentages = conf_matrix_percentages(confusion)
    print('\nConfusion Matrix (Test) in percentages\n', confusion_percentages)
    return train_uar, dev_uar, test_uar


def train_and_eval_models(train_x, test_x, train_y, test_y):
    train_x, dev_x, train_y, dev_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=42, stratify=train_y)
    train_x, scaler = data_utils.normalize(train_x)
    dev_x, _ = data_utils.normalize(dev_x, scaler=scaler)
    test_x, _ = data_utils.normalize(test_x, scaler=scaler)
    dump(scaler, 'scaler_{}.joblib'.format(MODALITY))
    cprint('Normalized data (zero-score standardization)', 'yellow')
    print('Dataset shapes: ', train_x.shape, dev_x.shape, test_x.shape)
    print('Class distributions for train/dev/test:')
    print(np.unique(train_y, return_counts=True))
    print(np.unique(dev_y, return_counts=True))
    print(np.unique(test_y, return_counts=True))

    # code for SVM training
#    svm = SVC(C=C, kernel=KERNEL, degree=DEGREE, gamma='auto')
#    svm.fit(train_x, train_y)
#    dump(svm, 'svm_{}_{}-kernel_C{}.joblib'.format(MODALITY, KERNEL, C))
    # code for MLP training
    mlp = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, alpha=ALPHA,
                        early_stopping=True, max_iter=200)
    mlp.fit(train_x, train_y)
    print(f'mlp.n_iter_: {mlp.n_iter_}')
    dump(mlp, 'mlp_{}.joblib'.format(MODALITY))
    # Decision Tree
#    dt = tree.DecisionTreeClassifier(max_depth=None)
#    dt.fit(train_x, train_y)
#    print('DT information\nDepth: {}\nNumber_leaves: {}'.format(
#        dt.get_depth(),
#        dt.get_n_leaves()
#    ))
#    dump(dt, 'dt_{}.joblib'.format(MODALITY))
    # print('DT feature importances:\n{}'.format(dt.feature_importances_))
    # print(dt.feature_importances_.shape, np.min(dt.feature_importances_),
    #       np.max(dt.feature_importances_), sum(dt.feature_importances_))

    # predictions
#    cprint('\nSVM PERFORMANCE', 'green')
#    cprint(f'SVM parameters:\n {svm}\n', 'yellow')
#    svm_uars = score(svm, train_x, train_y, dev_x, dev_y, test_x, test_y)
    cprint('\nMLP PERFORMANCE', 'green')
    cprint(f'MLP parameters:\n {mlp}\n', 'yellow')
    mlp_uars = score(mlp, train_x, train_y, dev_x, dev_y, test_x, test_y)
#    cprint('\nDT PERFORMANCE', 'green')
#    cprint(f'DT parameters:\n {dt}\n', 'yellow')
#    dt_uars = score(dt, train_x, train_y, dev_x, dev_y, test_x, test_y)
#    return svm_uars, mlp_uars, dt_uars
    return mlp_uars


if MODALITY == 'visual':
    visual_input = data_utils.InputData(VISUAL_DATA)
    data, class_lbls = get_data_and_labels(visual_input, LABEL_TYPE, LABELS)
elif MODALITY == 'audio':
    audio_input = data_utils.InputData(AUDIO_DATA)
    data, class_lbls = get_data_and_labels(audio_input, LABEL_TYPE, LABELS)
elif MODALITY == 'audiovisual':
    visual_input = data_utils.InputData(VISUAL_DATA)
    audio_input = data_utils.InputData(AUDIO_DATA)
    if LABEL_TYPE == 'category':
        idx_v = np.isin(visual_input.get_category(), list(LABELS.keys()))
        idx_a = np.isin(audio_input.get_category(), list(LABELS.keys()))
        data = pd.concat([audio_input.get_features()[idx_a],
                          visual_input.get_features()[idx_v]],
                         axis=1, join='inner')
        class_lbls = audio_input.get_category(data.index)
    elif LABEL_TYPE == 'arousal':
        data = pd.concat([audio_input.get_features(),
                          visual_input.get_features()],
                         axis=1, join='inner')
        class_lbls = group_continuous_labels(
            audio_input.get_arousal(data.index))
    elif LABEL_TYPE == 'valence':
        data = pd.concat([audio_input.get_features(),
                          visual_input.get_features()],
                         axis=1, join='inner')
        class_lbls = group_continuous_labels(
            audio_input.get_valence(data.index))

assert (class_lbls.index == data.index).all(),\
    'Data and class_lbls do not have the same index'


if ARGS.load_model:
    clf = load(ARGS.load_model)
    cprint('Model loaded from file', 'green')
    print(clf.get_params())
    # not reasonable to divide into train and dev set for scoring of loaded
    # model, but for now the simplest thing to use the score(...) function
    train_x, test_x, train_y, test_y = train_test_split(
        data, class_lbls, test_size=0.2, random_state=42, stratify=class_lbls)
    train_x, dev_x, train_y, dev_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y)
    score(clf, train_x, train_y, dev_x, dev_y, test_x, test_y)
else:
    if LOSO:
        svm_results = []
        mlp_results = []
        dt_results = []
        for gender, num in itertools.product(['M', 'F'], range(1,7)):
            speaker = gender + f'0{num}'
            cprint(f'CROSS VALIDATION FOLD {speaker} ...', 'green')
            test_x = data.filter(regex=f'-{speaker}-', axis=0)
            test_y = class_lbls.filter(regex=f'-{speaker}-', axis=0)
            train_x = data.filter(regex=f'MSP-IMPROV-....-(?!{speaker})', axis=0)
            train_y =  class_lbls.filter(regex=f'MSP-IMPROV-....-(?!{speaker})', axis=0)
#            svm_res, mlp_res, dt_res = train_and_eval_models(train_x, test_x, train_y, test_y)
            mlp_res = train_and_eval_models(train_x, test_x, train_y, test_y)
#            svm_results.append(svm_res)
#            mlp_results.append(mlp_res)
#            dt_results.append(dt_res)
#        print(f'\n\nSVM mean train/dev/test UARs: {np.mean(np.array(svm_results), axis=0)}')
        print(f'MLP mean train/dev/test UARs: {np.mean(np.array(mlp_results), axis=0)}')
#        print(f'DT mean train/dev/test UARs: {np.mean(np.array(dt_results), axis=0)}')
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            data, class_lbls, test_size=0.2, random_state=42, stratify=class_lbls)
        results = train_and_eval_models(train_x, test_x, train_y, test_y)
