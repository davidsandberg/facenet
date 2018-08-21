# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import pickle
import sys

import numpy as np
import progressbar as pb

from util.distance import get_distance


def find_threshold_sort(pos, neg):
    pos_list = sorted(pos, key=lambda x: x[0])
    neg_list = sorted(neg, key=lambda x: x[0], reverse=True)
    pos_count = len(pos_list)
    neg_count = len(neg_list)
    correct = 0
    threshold = 0
    # print('sort pos')
    # print(pos_list)
    # print('sort neg')
    # print(neg_list)
    for i in range(min(pos_count, neg_count)):
        if pos_list[i][0] > neg_list[i][0]:
            correct = i
            threshold = (pos_list[i][0] + neg_list[i][0]) / 2
            break
    # print("%d/%d" % (correct, pos_count))
    precision = (correct * 2.0) / (pos_count + neg_count)
    return precision, threshold


def get_accuracy(pos_list, neg_list, threshold):
    pos_count = len(pos_list)
    neg_count = len(neg_list)
    correct = 0
    for i in range(pos_count):
        if pos_list[i][0] < threshold:
            correct += 1

    for i in range(neg_count):
        if neg_list[i][0] > threshold:
            correct += 1
    precision = float(correct) / (pos_count + neg_count)
    return precision


def best_threshold(pos_list, neg_list, thrNum=10000):
    ts = np.linspace(-1, 1, thrNum * 2 + 1)
    best_acc = 0
    best_t = 0
    for t in ts:
        acc = get_accuracy(pos_list, neg_list, t)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_acc, best_t


def test_kfold(pos_list, neg_list, k=10):
    fold_size = len(pos_list) // k
    sum_acc = 0
    sum_thresh = 0
    sum_n = 0
    accu_list = []
    for i in range(k):
        val_pos = []
        val_neg = []
        test_pos = []
        test_neg = []
        for j in range(len(pos_list)):
            fi = j // fold_size
            if fi != i:
                val_pos.append(pos_list[j])
                val_neg.append(neg_list[j])
            else:
                test_pos.append(pos_list[j])
                test_neg.append(neg_list[j])
        precision, threshold = find_threshold_sort(val_pos, val_neg)
        accuracy = get_accuracy(test_pos, test_neg, threshold)
        accu_list.append(accuracy)
        sum_acc += accuracy
        sum_thresh += threshold
        sum_n += 1
        # verbose
        print('precision:%.4f threshold:%f' % (accuracy, threshold))
    return sum_acc / sum_n, sum_thresh / sum_n, accu_list


def compute_distance(pos_list, neg_list, dist_type='L2'):
    '''
    [
      [feat1, feat2, ..],
        ...
      [feat1, feat2, ..]
    ]
    '''
    # distance measure
    if isinstance(dist_type, str):
        dist_func = get_distance(dist_type)
    else:
        dist_func = dist_type
    # get dist
    pos_dist = []
    for i in pos_list:
        dist = dist_func(i[0], i[1])
        pos_dist.append([dist])

    neg_dist = []
    for i in neg_list:
        dist = dist_func(i[0], i[1])
        neg_dist.append([dist])
    return pos_dist, neg_dist


def verification(pos_list, neg_list, dist_type='L2'):
    '''
    [
      [feat1, feat2, ..],
        ...
      [feat1, feat2, ..]
    ]
    '''
    pos_dist, neg_dist = compute_distance(pos_list, neg_list, dist_type)
    precision, threshold, accu_list = test_kfold(pos_dist, neg_dist)
    pos = sorted(pos_dist, key=lambda x: x[0])
    neg = sorted(neg_dist, key=lambda x: x[0], reverse=True)
    pos = [x[0] for x in pos]
    neg = [x[0] for x in neg]
    acc, std = np.mean(accu_list), np.std(accu_list)

    return acc, std, threshold, pos, neg, accu_list


def extract_list_feature(extractor, pair_list, batch_size, size=0):
    feat_list = []
    npairs = len(pair_list)
    if size == 0:
        size = npairs * 2
    size = min(size, npairs * 2)
    num_batches = (size + batch_size - 1) // batch_size

    widgets = ['Batch Processing', pb.Percentage(), ' ',
               pb.Bar(marker=pb.Bar()), ' ', pb.ETA()]
    timer = pb.ProgressBar(
        widgets=widgets, max_value=int(num_batches + 1)).start()

    for batch_num in range(num_batches):
        # make a batch
        x_list = []
        for i in range(0, batch_size, 2):
            pairid = (batch_num * batch_size + i) // 2
            if pairid >= npairs:
                pairid = npairs - 1
            x_list.append(pair_list[pairid][0])
            x_list.append(pair_list[pairid][1])
        x_batch = np.stack(x_list, axis=0)
        feat = extractor.extract(x_batch)

        for i in range(0, batch_size, 2):
            a = feat[i, :]
            p = feat[i + 1, :]
            if len(feat_list) < size:
                feat_list.append([a, p])
        try:
            timer.update(batch_num)
        except BaseException:
            pass
    timer.finish()
    return feat_list
