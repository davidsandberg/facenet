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

class CenterLossTest(unittest.TestCase):
  


    def testCenterLoss(self):
        batch_size = 16
        nrof_features = 4
        alfa = 1.0
        
        with tf.Graph().as_default():
        
            logits = tf.placeholder(tf.float32, shape=(batch_size, nrof_features), name='logits')
            labels = tf.placeholder(tf.int32, shape=(batch_size,), name='labels')
            centers = tf.get_variable('centers', shape=(nrof_features), dtype=tf.float32,
                initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
            # Define center loss
            center_loss = tf.reduce_sum(tf.pow(tf.abs(logits - centers), 2.0))
            one_hot = tf.one_hot(labels, nrof_features, axis=1, dtype=tf.float32, name='one_hot')
            centers_delta = tf.reduce_mean((centers-logits)*one_hot,0) / (1+tf.reduce_mean(one_hot,0))
            update_centers = tf.assign(centers, tf.add(centers, -alfa*centers_delta))
                
            sess = tf.Session()
            with sess.as_default():
                sess.run(tf.initialize_all_variables())
                np.random.seed(seed=666)
                x = np.transpose(np.matmul(np.expand_dims(np.arange(0.1,0.5,0.1),1), np.ones(shape=(1, batch_size))))

                #center_loss_, centers_, one_hot_, fx_, num_, den_ = sess.run([center_loss, centers, one_hot, fx, num, den], feed_dict={logits:x, labels:y})
                for i in range(0,50):
                    cls = i % nrof_features
                    #y = np.ones(shape=(batch_size), dtype=np.float32) * cls
                    y = np.zeros(shape=(batch_size), dtype=np.float32)
                    y[:batch_size/2] = i % nrof_features
                    y[batch_size/2:] = (i+2) % nrof_features
                    center_loss_, centers_ = sess.run([center_loss, centers], feed_dict={logits:x, labels:y})
                    print(center_loss_)
                    print(centers_)
                    print('')
                    _ = sess.run(update_centers, feed_dict={logits:x, labels:y})
                    center_loss_, centers_ = sess.run([center_loss, centers], feed_dict={logits:x, labels:y})
                    print(center_loss_)
                    print(centers_)
                    print('')
                    xxx = 1
                
                      
if __name__ == "__main__":
    unittest.main()
