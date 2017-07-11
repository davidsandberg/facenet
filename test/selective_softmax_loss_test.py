# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
import tensorflow as tf
import numpy as np
import facenet

class SelectiveSoftmaxLossTest(unittest.TestCase):

    def testCrossEntropySelection(self):
        batch_size = 2
        nrof_classes = 5
        use_label_probabilities = False
        
        with tf.Graph().as_default():
        
            logits = tf.placeholder(tf.float32, shape=(batch_size, nrof_classes), name='logits')
            labels = tf.placeholder(tf.int32, shape=(batch_size,), name='labels')
            class_thresholds = tf.placeholder(tf.float32, shape=(batch_size,), name='class_thresholds')
            cross_entropy_selected, max_class, max_prob = facenet.selective_softmax_loss(
                logits, labels, nrof_classes, class_thresholds, use_label_probabilities)
                
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                lg = np.array([[0.1, 0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.9]])
                lb = np.array([2, 4])
                cth = np.array([0.32, 0.32])
                cross_entropy_selected_, max_class_, max_prob_ = sess.run(
                    [cross_entropy_selected, max_class, max_prob], feed_dict={logits:lg, labels:lb, class_thresholds:cth})
                #print(prob)
                #print(cross_entropy_)
                print(cross_entropy_selected_)
                print(max_class_)
                print(max_prob_)
                
                #np.testing.assert_almost_equal(tf_triplet_loss, np_triplet_loss, decimal=5, err_msg='Triplet loss is incorrect')
                
if __name__ == "__main__":
    unittest.main()
