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
import models
import numpy as np
import numpy.testing as testing

class BatchNormTest(unittest.TestCase):


    @unittest.skip("Skip batch norm test case")
    def testBatchNorm(self):
      
        tf.set_random_seed(123)
  
        x = tf.placeholder(tf.float32, [None, 20, 20, 10], name='input')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        # generate random noise to pass into batch norm
        #x_gen = tf.random_normal([50,20,20,10])
        
        bn = models.network.batch_norm(x, phase_train)
        
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)
  
        with sess.as_default():
        
            #generate a constant variable to pass into batch norm
            y = np.random.normal(0, 1, size=(50,20,20,10))
            
            feed_dict = {x: y, phase_train: True}
            sess.run(bn, feed_dict=feed_dict)
            
            feed_dict = {x: y, phase_train: False}
            y1 = sess.run(bn, feed_dict=feed_dict)
            y2 = sess.run(bn, feed_dict=feed_dict)
            
            testing.assert_almost_equal(y1, y2, 10, 'Output from two forward passes with phase_train==false should be equal')


if __name__ == "__main__":
    unittest.main()
    