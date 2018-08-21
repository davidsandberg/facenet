# -*- coding:utf-8 -*-
from __future__ import print_function

import io
import math
import os
import random
import sys
import time

import cv2
import mxnet as mx
import numpy as np
import PIL.Image
from PIL import Image


def read_idx(p):
    s = p[0]
    label = p[1]
    header, img = mx.recordio.unpack(s)
    try:
        label_ = int(header.label)
    except Exception as e:
        print(header.label)
    # assert label == label_
    # img, label = p[0], p[1]
    encoded_jpg_io = io.BytesIO(img)
    image = PIL.Image.open(encoded_jpg_io)
    np_img = np.array(image)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return img, label


class MxReader:
    def __init__(self, datadir, balance=False, name='ms1m'):
        self.setname = name
        self.datadir = datadir
        self.balance = balance

        idx_path = os.path.join(datadir, 'train.idx')
        bin_path = os.path.join(datadir, 'train.rec')
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        assert header.flag > 0
        max_index = int(header.label[0])
        min_seq_id = max_index
        max_seq_id = int(header.label[1])
        identities = max_seq_id - min_seq_id
        id2range = []
        for id in range(identities):
            identity = id + min_seq_id
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            a, b = int(header.label[0]), int(header.label[1])
            size = b - a
            id2range.append((a, b, size))
        '''
        print(max_index)
        print(max_seq_id)
        print(id2range[0])
        print(id2range[1])
        print(id2range[-1])
        '''
        self.imgrec = imgrec
        self.images = max_index - 1
        self.identities = identities
        self.id2range = id2range
        self.cbuf = [i for i in range(identities)]
        self.cid = 0
        self.list = [i for i in range(1, max_index)]
        self.cursor = 0
        self.id_offset = 0
        # shuffle
        self.digest()

    def setBalance(self, balance):
        self.balance = balance

    def name(self):
        return self.setname

    def add(self, line):
        pass

    def digest(self):
        random.shuffle(self.cbuf)
        self.cid = 0
        random.shuffle(self.list)
        self.cursor = 0

    def verbose(self):
        print('Dataset:%8s size:%6d nclass:%d' %
              (self.setname, self.size(), self.numOfClass()))

    def size(self):
        return self.images

    def numOfClass(self):
        return self.identities

    def maxClass(self):
        return self.identities - 1

    def minClass(self):
        return 0

    def findLabel(self, index):
        low = 0
        high = len(self.id2range)
        while low <= high:
            mid = (low + high) / 2
            y = self.id2range[mid]
            if index >= y[0] and index < y[1]:
                return mid
            elif index >= y[1]:
                low = mid + 1
            else:
                high = mid - 1

    def getSample(self, index):
        if self.balance:
            # balanced
            if self.cid >= self.identities:
                random.shuffle(self.cbuf)
                self.cid = 0
            label = self.cbuf[self.cid]
            a, b, _ = self.id2range[label]
            index = random.randint(a, b - 1)
            self.cid += 1
        else:
            index = self.list[index]
            # index to label
            label = self.findLabel(index)

        return index, label

    def next(self):
        '''
        Return audio path, and label
        '''
        index = self.cursor
        self.cursor += 1
        self.cursor %= self.size()
        return self.getSample(index)

    def nextTask(self):
        index, label = self.next()
        s = self.imgrec.read_idx(index)
        return (read_idx, (s, label))

    def moreTask(self, y):
        a, b, _ = self.id2range[y]
        index = random.randint(a, b - 1)
        s = self.imgrec.read_idx(index)
        return (read_idx, (s, y))

    def close(self):
        self.imgrec.close()
        print("%s closed" % (self.name()))


if __name__ == '__main__':
    ms1m = MxReader('/home/ysten/data/insightface/faces_ms1m_112x112')
    ms1m.verbose()
    s = ms1m.imgrec.read_idx(1)
    read_idx((s, 0))
    s = ms1m.imgrec.read_idx(ms1m.list[-1])
    read_idx((s, 0))
    s = ms1m.imgrec.read_idx(ms1m.size())
    read_idx((s, 0))
    exit()
    for i in range(10000):
        func, param = ms1m.nextTask()
        img, label = func(param)
        print("%d label:%d [%d, %d]" % (i, label, img.shape[0], img.shape[1]))
