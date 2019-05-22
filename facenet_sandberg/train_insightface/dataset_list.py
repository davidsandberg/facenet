# -*- coding:utf-8 -*-
from __future__ import print_function
import scipy
import numpy as np
import os
import sys
import time
import random
import math
import cv2


def read_img(p):
    path, y = p[0], p[1]
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x, y


class FilelistReader:
    def __init__(self, datadir, balance=False, name='--'):
        self.setname = name
        self.datadir = datadir
        self.celeb = []
        self.dict = {}
        self.list = []
        self.balance = balance
        self.cursor = 0
        self.cid = 0
        self.cbuf = []
        self.id_offset = 0

    def setBalance(self, balance):
        self.balance = balance

    def name(self):
        return self.setname

    def add(self, line):
        line = line.strip()
        if line.find(',') > 0:
            segs = line.split(',')
        else:
            segs = line.split()
        # print(segs)
        if len(segs) != 2:
            return False
        rel_path = segs[0]
        label = int(segs[1])
        item = (rel_path, label)
        # add to list
        self.list.append(item)

    def digest(self):
        random.shuffle(self.list)
        self.dict = {}
        self.cursor = 0
        for index in range(len(self.list)):
            item = self.list[index]
            label = item[1]
            # add to dict[label]
            if label not in self.dict.keys():
                self.dict[label] = []
            self.dict[label].append(index)
        # update celeb
        # print(self.dict)
        keys = sorted(self.dict.keys())
        self.celeb = keys
        self.cid = 0
        self.cbuf = [i for i in range(len(self.celeb))]
        # random.shuffle(self.cbuf)

    def verbose(self):
        print('Dataset:%8s size:%6d nclass:%d' %
              (self.setname, self.size(), self.numOfClass()))

    def size(self):
        return len(self.list)

    def numOfClass(self):
        return len(self.celeb)

    def maxClass(self):
        return max(self.celeb)

    def minClass(self):
        return min(self.celeb)

    def getSample(self, index):
        if self.balance:
            # balanced
            if self.cid >= len(self.celeb):
                random.shuffle(self.cbuf)
                self.cid = 0
            label = self.celeb[self.cbuf[self.cid]]
            index = np.random.choice(self.dict[label])
            self.cid += 1
        item = self.list[index]
        rel_path = item[0]
        path = os.path.join(self.datadir, rel_path)
        return path, item[1]

    def next(self):
        '''
        Return audio path, and label
        '''
        index = self.cursor
        self.cursor += 1
        self.cursor %= self.size()
        return self.getSample(index)

    def nextTask(self):
        path, y = self.next()
        return (read_img, (path, y))

    def moreTask(self, y):
        index = np.random.choice(self.dict[y])
        item = self.list[index]
        rel_path = item[0]
        path = os.path.join(self.datadir, rel_path)
        return (read_img, (path, y))

    def close(self):
        self.celeb = []
        self.dict = {}
        self.list = []
        print("%s closed" % (self.name()))


if __name__ == '__main__':
    pass
