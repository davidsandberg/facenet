import unittest
import tensorflow as tf
import facenet
import numpy as np
import numpy.testing as testing

class BatchNormTest(unittest.TestCase):


    def testBatchNorm(self):
      
        tf.set_random_seed(123)
  
        x = tf.placeholder(tf.float32, [None, 20, 20, 10], name='input')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        # generate random noise to pass into batch norm
        #x_gen = tf.random_normal([50,20,20,10])
        
        bn = facenet.batch_norm(x, 10, phase_train, 'batch_norm', affine=True)
        
        init = tf.initialize_all_variables()
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
    