# -*- coding:utf-8 -*-
import math
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class TensorflowExtractor:
    def __init__(self, sess, embedding_tensor, batch_size,
                 feed_dict, input_placeholder):
        # save context
        self.embedding_tensor = embedding_tensor
        self.batch_size = batch_size
        self.sess = sess
        self.feed_dict = feed_dict
        self.input_placeholder = input_placeholder

    def extract(self, x_batch: np.ndarray):
        self.feed_dict.setdefault(self.input_placeholder, None)
        self.feed_dict[self.input_placeholder] = x_batch
        feat = self.sess.run(self.embedding_tensor, feed_dict=self.feed_dict)
        feat = np.array(feat)
        feat = np.squeeze(feat)
        return feat

    def close(self):
        self.sess.close()
