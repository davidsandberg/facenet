import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as testing

class DecovLossTest(unittest.TestCase):


    def testDecovLoss(self):
        batch_size = 7
        image_size = 4
        channels = 3
        
        with tf.Graph().as_default():
        
            xs = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, channels), name='input')
            x = tf.reshape(xs, [batch_size, image_size*image_size*channels])
            
            m = tf.reduce_mean(x, 0, True)
            z = tf.expand_dims(x-m, 2)
            corr = tf.reduce_mean(tf.batch_matmul(z, tf.transpose(z, perm=[0,2,1])), 0)
            corr_frob_sqr = tf.reduce_sum(tf.square(corr))
            corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
            loss = 0.5*(corr_frob_sqr - corr_diag_sqr)
      
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
