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
#import facenet

class SelectiveSoftmaxLossTest(unittest.TestCase):

    def testSelectiveSoftmaxLoss(self):
        batch_size = 2
        nrof_classes = 5
        
        with tf.Graph().as_default():
        
            logits = tf.placeholder(tf.float32, shape=(batch_size, nrof_classes), name='logits')
            labels = tf.placeholder(tf.int32, shape=(batch_size,), name='labels')
            class_thresholds = tf.placeholder(tf.float32, shape=(batch_size,), name='class_thresholds')
            cross_entropy, max_prob, label_prob = selective_softmax_loss(logits, labels, nrof_classes, class_thresholds)
                
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                lg = np.array([[0.1, 0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.9]])
                lb = np.array([2, 4])
                cth = np.array([0.3, 0.3])
                cross_entropy_, max_prob_, label_prob_ = sess.run([cross_entropy, max_prob, label_prob], feed_dict={logits:lg, labels:lb, class_thresholds:cth})
                #print(prob)
                print(cross_entropy_)
                print(max_prob_)
                print(label_prob_)
                
                #np.testing.assert_almost_equal(tf_triplet_loss, np_triplet_loss, decimal=5, err_msg='Triplet loss is incorrect')
                      
def selective_softmax_loss(logits, labels, nrof_classes, class_thresholds_for_batch):
    #indices = tf.stack((tf.range(tf.shape(prob)[0], dtype=tf.int64), max_class), axis=1)
    labels_onehot = tf.one_hot(labels, nrof_classes, on_value=1.0, off_value=0.0, axis=1, dtype=tf.float32)
    prob = tf.nn.softmax(logits)
    max_class = tf.argmax(prob, axis=1)
    cross_entropy = -tf.reduce_sum(labels_onehot * tf.log(prob), 1)
    max_indices = tf.stack((tf.range(prob.get_shape()[0], dtype=max_class.dtype), max_class), axis=1)
    max_prob = tf.gather_nd(prob, max_indices)
    
    label_indices = tf.stack((tf.range(prob.get_shape()[0], dtype=labels.dtype), labels), axis=1)
    label_prob = tf.gather_nd(prob, label_indices)

    # One probability threshold per class
    # Update probability threshold every step
    # Set cross_entropy for the example to zero if probability < threshold
    #   - how to get threshold: could be from label or max_class
    
    return cross_entropy, max_prob, label_prob

if __name__ == "__main__":
    unittest.main()
