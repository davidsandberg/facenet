
import sys
import time

import numpy as np

from config import Config
from dataset_list import FilelistReader
from dataset_multi import MultiDataset
from dataset_mxnet import MxReader
from mt_loader import MultiThreadLoader


def get_WebFace(config, count=-1):
    ds = FilelistReader(config.get('WebFace').prefix, name='WebFace')
    with open(config.get('WebFace').train_list, 'r') as f:
        idx = 0
        for line in f.readlines():
            ds.add(line)
            idx += 1
            if idx == count:
                break

    ds.digest()
    return ds


def get_ms1m(config):
    ds = MxReader(config.get('ms1m').mxrec, name='ms1m')
    ds.digest()
    return ds


def get_vgg(config):
    ds = MxReader(config.get('vgg').mxrec, name='vgg')
    ds.digest()
    return ds


def dataset_factory(name, config):
    '''
    surport ms1m vgg WebFace YTF
    '''
    dict = {
        'WebFace': get_WebFace,
        'ms1m': get_ms1m,
        'vgg': get_vgg,
    }
    return dict[name](config)


def build_dataset(config, list, balance=False, num_per_class=1):
    ds = MultiDataset(balance, num_per_class=num_per_class)
    for i in list:
        name, weight = i[0], i[1]
        ds_ = dataset_factory(name, config)
        ds_.setBalance(balance)
        ds.add(ds_, weight)
    ds.verbose()
    return ds


def test_ds(dg):
    for i in range(10000):
        func, param = dg.nextTask()
        img, label = func(param)
        print("%d label:%d [%d, %d]" % (i, label, img.shape[0], img.shape[1]))


def test_db():
    dataset_list = []
    dataset_list.append(('vgg', -1))
    # dataset_list.append(('ms1m',    -1))
    dataset_list.append(('WebFace', -1))
    dataset = build_dataset(dataset_list, balance=True, num_per_class=4)
    for i in range(1000000):
        start = time.time()
        func, param = dataset.nextTask()
        img, label = func(param)
        # x, y = dg.getBatch()
        end = time.time()
        # plot_fbank_cm(fbank_feat)
        # print("x.shape:{0}, y.shape:{1}".format(x.shape, y.shape))
        print(label)
        if i % 1000 == 0:
            print('batch:%d t:%f ' % (i, end - start))

        # print('t:%f' % (end - start) )


if __name__ == '__main__':
    test_db()
    exit()
    list_ = []
    list_.append(('vgg', 1))
    list_.append(('ms1m', -1))
    dataset = build_dataset(list_)
    # test_ds(dataset)
    dg = MultiThreadLoader(dataset, 5, 1)
    print(dg.numOfClass())
    for i in range(100):
        print('batch:%d' % (i))
        start = time.time()
        x, y = dg.getBatch()
        end = time.time()
        # plot_fbank_cm(fbank_feat)
        # print("x.shape:{0}, y.shape:{1}".format(x.shape, y.shape))
        print(y)
        # print('t:%f' % (end - start) )
    dg.close()
    # sys.exit()
