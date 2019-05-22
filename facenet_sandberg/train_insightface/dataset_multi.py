import random
import time

import numpy as np


class MultiDataset:
    def __init__(self, balance=False, num_per_class=1):
        self.balance = balance
        self.num_per_class = num_per_class
        self.last_ds = None
        self.last_y = -1
        self.last_n = 0
        self.datasets = []
        self.weights_ = []
        self.weights = []
        self.totalClass = 0
        self.totalSize = 0

    def add(self, dataset, weight=-1):
        self.datasets.append(dataset)
        dataset.id_offset = self.totalClass
        if weight < 0:
            if self.balance:
                weight = dataset.numOfClass()
            else:
                weight = dataset.size()

        self.weights_.append(weight)
        self._normWeights()
        self.totalClass += dataset.numOfClass()
        self.totalSize += dataset.size()

    def _normWeights(self):
        sw = float(sum(self.weights_))
        self.weights = [i / sw for i in self.weights_]

    def digest(self):
        _normWeights()

    def size(self):
        return self.totalSize

    def numOfClass(self):
        return self.totalClass

    def maxClass(self):
        return self.totalClass - 1

    def minClass(self):
        return 0

    def verbose(self):
        for i in range(len(self.datasets)):
            dataset = self.datasets[i]
            print(
                'Dataset:%8s size:%6d nclass:%6d  %.3f' %
                (dataset.name(),
                 dataset.size(),
                 dataset.numOfClass(),
                 self.weights[i]))
        print('------------------------------------------------------------')
        print('Dataset:%8s size:%6d nclass:%6d' %
              ('total', self.size(), self.numOfClass()))

    def nextTask(self):
        if self.last_y < 0 or self.last_n >= self.num_per_class:
            self.last_y = -1
            # next class
            ds = np.random.choice(self.datasets, p=self.weights)
            func, param = ds.nextTask()
            lp = list(param)
            # y
            self.last_y = lp[1]
            self.last_n = 1
            self.last_ds = ds
            lp[1] += ds.id_offset
            param = tuple(lp)
            return (func, param)
        else:
            func, param = self.last_ds.moreTask(self.last_y)
            lp = list(param)
            # y
            self.last_n += 1
            lp[1] += self.last_ds.id_offset
            param = tuple(lp)
            return (func, param)

    def close(self):
        for ds in self.datasets:
            ds.close()
        self.datasets = []


if __name__ == '__main__':
    '''
    y = [1,2,3,0]
    y_train = np_utils.to_categorical(y, 4)
    print(y_train)
    exit()
    '''
    dg = DataGeneratorMT(32, 1)
    print(dg.numOfClass())
    for i in range(100000):
        print('batch:%d' % (i))
        start = time.time()
        x, y = dg.getBatch()
        end = time.time()
        # plot_fbank_cm(fbank_feat)
        # print("x.shape:{0}, y.shape:{1}".format(x.shape, y.shape))
        print(y)
        # print('t:%f' % (end - start) )
    dg.close()
