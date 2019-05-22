
import logging
import random
import sys
import time
from multiprocessing import Process, Queue, Value

import numpy as np


def transform_mirror(x):
    if random.random() < 0.5:
        x = np.fliplr(x)
    return x


def crop_image(img1, imsize):
    h, w, c = img1.shape
    x1 = (w - imsize[0]) / 2
    y1 = (h - imsize[1]) / 2
    img1_ = img1[y1:(y1 + imsize[1]), x1:(x1 + imsize[0]), :]
    return img1_


def transform_crop_96x112(x):
    return crop_image(x, (96, 112))


transforms = [transform_crop_96x112]


def addTransform(self, func):
    global transforms
    if func not in transforms:
        transforms.append(func)


def threadProc(todo, done, quit_signal):
    global transforms
    while quit_signal != 1:
        try:
            task = todo.get()
            func = task[0]
            param = task[1]
            start = time.time()
            x, y = func(param)
            # do transform
            for t in transforms:
                x = t(x)
            if quit_signal == 1:
                break
            done.put((x, y))
            # print("done id:%d" % y)
        except Exception as e:
            # time.sleep(0.5)
            # print(task)
            print(e)
            sys.exit(0)


class MultiThreadLoader:
    def __init__(self, dataset, batch_size, nworkers=1):
        self.B = dataset
        self.batch_size = batch_size
        # todo list
        self.maxsize = batch_size * 2
        self.todo = Queue(self.maxsize)
        # done list
        self.done = Queue(self.maxsize)
        # create threads
        self.feed()
        self.quit_signal = Value('i', 0)
        self.createThread(nworkers)

    def numOfClass(self):
        return self.B.numOfClass()

    def size(self):
        return self.B.size()

    def shuffle(self):
        # shuffle
        self.B.digest()
        # prefeed
        self.feed()

    def createThread(self, nworkers):
        self.threads = []
        # self.db_lock = threading.Lock()
        for i in range(nworkers):
            t = Process(
                target=threadProc,
                args=(
                    self.todo,
                    self.done,
                    self.quit_signal),
                name='worker/' +
                str(i))
            t.start()
            self.threads.append(t)

    def feed(self):
        if self.todo.full():
            return
        n = self.maxsize - self.todo.qsize()
        for i in range(n):
            task = self.B.nextTask()
            self.todo.put(task)
            # print("todo id:%d" % y)

    def fetch(self):
        x_list = []
        y_list = []
        for i in range(self.batch_size):
            x, y = self.done.get()
            # print("fetch id:%d" % y)
            x_list.append(x)
            y_list.append(y)
        x_batch = np.stack(x_list, axis=0)
        y_batch = np.array(y_list)
        # x_batch = np.transpose(x_batch,[0,2,1,3])
        return x_batch, y_batch

    def getBatch(self):
        start = time.time()
        ret = self.fetch()
        end = time.time()
        self.feed()
        t2 = time.time()
        # print('fetch:%f feed:%f' % (end - start, t2 - end) )
        return ret

    def close(self):
        self.quit_signal = 1
        print("mtloader close")

        for t in self.threads:
            try:
                t.terminate()
                t.process.signal(signal.SIGINT)
            except BaseException:
                pass
        for t in self.threads:
            print(t.is_alive())

        self.threads = []
        # close datasets
        self.B.close()
