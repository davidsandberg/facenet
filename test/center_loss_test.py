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

class CenterLossTest(unittest.TestCase):
  


    def testCenterLoss(self):
        batch_size = 16
        nrof_features = 2
        nrof_classes = 16
        alfa = 0.5
        
        with tf.Graph().as_default():
        
            features = tf.placeholder(tf.float32, shape=(batch_size, nrof_features), name='features')
            labels = tf.placeholder(tf.int32, shape=(batch_size,), name='labels')

            # Define center loss
            center_loss, centers = facenet.center_loss(features, labels, alfa, nrof_classes)
            
            label_to_center = np.array( [ 
                 [-3,-3],  [-3,-1],  [-3,1],  [-3,3],
                 [-1,-3],  [-1,-1],  [-1,1],  [-1,3],
                 [ 1,-3],  [ 1,-1],  [ 1,1],  [ 1,3],
                 [ 3,-3],  [ 3,-1],  [ 3,1],  [ 3,3] 
                 ])
                
            sess = tf.Session()
            with sess.as_default():
                sess.run(tf.global_variables_initializer())
                np.random.seed(seed=666)
                
                for _ in range(0,100):
                    # Create array of random labels
                    lbls = np.random.randint(low=0, high=nrof_classes, size=(batch_size,))
                    feats = create_features(label_to_center, batch_size, nrof_features, lbls)

                    center_loss_, centers_ = sess.run([center_loss, centers], feed_dict={features:feats, labels:lbls})
                    
                # After a large number of updates the estimated centers should be close to the true ones
                np.testing.assert_almost_equal(centers_, label_to_center, decimal=5, err_msg='Incorrect estimated centers')
                np.testing.assert_almost_equal(center_loss_, 0.0, decimal=5, err_msg='Incorrect center loss')
                

def create_features(label_to_center, batch_size, nrof_features, labels):
    # Map label to center
#     label_to_center_dict = { 
#          0:(-3,-3),  1:(-3,-1),  2:(-3,1),  3:(-3,3),
#          4:(-1,-3),  5:(-1,-1),  6:(-1,1),  7:(-1,3),
#          8:( 1,-3),  9:( 1,-1), 10:( 1,1), 11:( 1,3),
#         12:( 3,-3), 13:( 3,-1), 14:( 3,1), 15:( 3,3),
#         }
    # Create array of features corresponding to the labels
    feats = np.zeros((batch_size, nrof_features))
    for i in range(batch_size):
        cntr =  label_to_center[labels[i]]
        for j in range(nrof_features):
            feats[i,j] = cntr[j]
    return feats
                      
if __name__ == "__main__":
    unittest.main()
