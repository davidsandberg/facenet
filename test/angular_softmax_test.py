# MIT License
# 
# Copyright (c) 2017 David Sandberg
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
import math
import facenet

class AngularSoftmaxLossTest(unittest.TestCase):

    def testAngularSoftmax(self):
        batch_size = 2
        nrof_classes = 4
        nrof_features = 3
        
        with tf.Graph().as_default():
        
            weights = tf.placeholder(tf.float32, shape=(nrof_features, nrof_classes), name='weights')
            features = tf.placeholder(tf.float32, shape=(batch_size, nrof_features), name='features')
            labels = tf.placeholder(tf.int32, shape=(batch_size,), name='labels')
            loss = facenet.angular_softmax_loss(weights, features, labels, 4)
            
            sess = tf.Session()
            with sess.as_default():
                np.random.seed(seed=666)
                
                x = np.array([[1, 1, 0], [2, 2, 1]])#.transpose()
                yi = np.full((2,), [2, 1])
                W = np.array([[ 1, 1, 0 ],[ 0.75, 0.25, 0 ],[ 0.5, 0.5, 0 ],[ 0.25, 0.75, 0]]).transpose()
                
                # Calculate reference loss (using python, numpy and for-loops)
                loss_ref = angular_softmax_loss_ref(W, x, yi, 4)

                # Check vectorized numpy implementation
                loss_numpy = angular_softmax_loss_np(W, x, yi, 4)
                np.testing.assert_array_almost_equal(loss_numpy, loss_ref, decimal=5, err_msg='Numpy vectorized loss does not match reference')
                
                # Check tensorflow implementation
                loss_ = sess.run(loss, feed_dict={weights:W, features:x, labels:yi})
                
                np.testing.assert_array_almost_equal(loss_, loss_ref, decimal=5, err_msg='Tensorflow loss does not match reference')
                
def angular_softmax_loss_ref(W, x, yi, m):
    nrof_classes = W.shape[1]
    batch_size = x.shape[0]
    
    theta = np.zeros((batch_size, nrof_classes))
    for i in range(batch_size):
        for j in range(nrof_classes):
            theta[i,j] = angular_difference_np(x[i,:], W[:,j]);

    Lang = np.zeros((batch_size,));
    for i in range(batch_size):
        nx = np.linalg.norm(x[i,:])
        qi = np.exp(nx*mod_cosine_np(theta[i,yi[i]], m))
        num = qi
        den = 0.0
        for j in range(nrof_classes):
            if j != yi[i]:
                #den += np.exp(np.linalg.norm(x[i,yi[i]])*np.cos(theta[i,j]));
                den += np.exp(nx*np.cos(theta[i,j]));
        Lang[i] = -np.log(num / (qi+den));
    return Lang
  
def angular_difference_np(a, b):
#         % result should be in the range 0..pi
    theta = np.arccos(dot_np(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)));
    return theta
  
def mod_cosine_np(theta, m):
    k = np.floor(theta*m/math.pi);
    y = np.power((-1), k) * np.cos(m*theta) - 2*k;
    return y
      
def angular_softmax_loss_np(W, x, yi, m):
    nrof_classes = W.shape[1]
    batch_size = x.shape[0]
    theta = angular_difference_matrix_np(x, W)
    nx = np.linalg.norm(x, axis=1)
    bi = np.arange(batch_size)
    theta_yi = theta[bi, yi]
    mct = mod_cosine_np(theta_yi, m)
    qi = np.exp(nx * mct)
    denx = np.exp(np.expand_dims(nx, axis=1) * np.cos(theta))
    j = np.expand_dims(np.arange(nrof_classes), axis=0)
    yi_exp = np.expand_dims(yi,1)
    den = np.sum(np.where(yi_exp!=j, denx, np.zeros_like(denx, dtype=np.float32)), 1)
    loss = -np.log(np.divide(qi, qi+den))
    return loss
                
def angular_difference_matrix_np(x, W):
    nrof_classes = W.shape[1]
    batch_size = x.shape[0]
    dot = np.tensordot(x, W, axes=1)
    na = np.expand_dims(np.linalg.norm(x, 2, axis=1), 1) * np.ones((1,nrof_classes))
    nb = np.ones((batch_size,1)) * np.expand_dims(np.linalg.norm(W, 2, axis=0), 0)
    xp = np.divide(dot, np.multiply(na, nb))
    res = np.arccos(xp)
    return res

def dot_np(a, b):
    return np.sum(a * b)
  
                
if __name__ == "__main__":
    unittest.main()
