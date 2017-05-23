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

class TripletLossTestNew(unittest.TestCase):
  
    def testBatchHard(self):
        emb = np.transpose(np.array([[ 1.0,  2.0,  3.0,   5.0,  6.0,  7.0 ]] ))

        P = 2  # Number of classes
        K = 3  # Number of images per class
        batch_size = P*K
        nrof_features = 1
        m = 0.2
        
        with tf.Graph().as_default():
        
            embeddings = tf.placeholder(tf.float32, shape=(batch_size, nrof_features), name='embeddings')
            loss, active_triplets_fraction = facenet.batch_hard_triplet_loss(embeddings, m, P, K, False)
            
            sess = tf.Session()
            with sess.as_default():
                sess.run(tf.global_variables_initializer())

                loss_, acf_ = sess.run([ loss, active_triplets_fraction ], feed_dict={embeddings:emb})
            unittest.TestCase.assertAlmostEqual(self, loss_, 0.0666666)
            unittest.TestCase.assertAlmostEqual(self, acf_, 1.0/3)
                
if __name__ == "__main__":
    unittest.main()
