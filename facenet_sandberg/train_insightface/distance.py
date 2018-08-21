# -*- coding:utf-8 -*-
import math
import numpy as np


def cosine_similarity(v1, v2):
    # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def cosine_distance(v1, v2):
    return 1 - cosine_similarity(v1, v2)


def L2_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


def SSD_distance(v1, v2):
    return np.sum(np.square(v1 - v2))


def get_distance(dist_type):
    loss_map = {
        'cosine': cosine_distance,
        'L2': L2_distance,
        'SSD': SSD_distance}
    return loss_map[dist_type]
