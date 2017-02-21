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

class DemuxEmbeddingsTest(unittest.TestCase):
  
    def testDemuxEmbeddings(self):
        batch_size = 3*12
        embedding_size = 16
        alpha = 0.2
        
        with tf.Graph().as_default():
        
            embeddings = tf.placeholder(tf.float64, shape=(batch_size, embedding_size), name='embeddings')
            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,embedding_size]), 3, 1)
            triplet_loss = facenet.triplet_loss(anchor, positive, negative, alpha)
                
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                emb = np.random.uniform(size=(batch_size, embedding_size))
                tf_triplet_loss = sess.run(triplet_loss, feed_dict={embeddings:emb})

                pos_dist_sqr = np.sum(np.square(emb[0::3,:]-emb[1::3,:]),1)
                neg_dist_sqr = np.sum(np.square(emb[0::3,:]-emb[2::3,:]),1)
                np_triplet_loss = np.mean(np.maximum(0.0, pos_dist_sqr - neg_dist_sqr + alpha))
                
                np.testing.assert_almost_equal(tf_triplet_loss, np_triplet_loss, decimal=5, err_msg='Triplet loss is incorrect')
                      
if __name__ == "__main__":
    unittest.main()
