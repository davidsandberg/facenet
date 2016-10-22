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
import numpy.testing as testing
import facenet

class DecovLossTest(unittest.TestCase):

    def testDecovLoss(self):
        batch_size = 7
        image_size = 4
        channels = 3
        
        with tf.Graph().as_default():
        
            xs = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, channels), name='input')
            loss = facenet.decov_loss(xs)
      
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                xs_ = np.random.normal(loc=0.0, scale=0.1, size=(batch_size,image_size,image_size,channels))
                xflat = xs_.reshape([batch_size,image_size*image_size*channels])
                ui = np.mean(xflat,0)
                nd = image_size*image_size*channels
                corr_ref = np.zeros((nd,nd))
                for i in range(nd):
                    for j in range(nd):
                        corr_ref[i,j] = 0.0
                        for n in range(batch_size):
                            corr_ref[i,j] += (xflat[n,i]-ui[i]) * (xflat[n,j]-ui[j]) / batch_size
                
                corr_frob_sqr_ref = np.trace(np.matmul(corr_ref.T, corr_ref))
                corr_diag_sqr_ref = np.sum(np.square(np.diag(corr_ref)))
                loss_ref = 0.5*(corr_frob_sqr_ref - corr_diag_sqr_ref)
                
                loss_ = sess.run(loss, feed_dict={xs:xs_})
                
                testing.assert_almost_equal(loss_ref, loss_, 6, 
                    'Tensorflow implementation gives a different result compared to reference')
      
if __name__ == "__main__":
    unittest.main()
