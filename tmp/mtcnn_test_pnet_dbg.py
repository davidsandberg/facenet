from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as io
import align.detect_face

#ref = io.loadmat('pnet_dbg.mat')
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        with tf.variable_scope('pnet'):
#            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            data = tf.placeholder(tf.float32, (1,1610, 1901,3), 'input')
            pnet = align.detect_face.PNet({'data':data})
            pnet.load('../../data/det1.npy', sess)
#         with tf.variable_scope('rnet'):
#             data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
#             rnet = align.detect_face.RNet({'data':data})
#             rnet.load('../../data/det2.npy', sess)
#         with tf.variable_scope('onet'):
#             data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
#             onet = align.detect_face.ONet({'data':data})
#             onet.load('../../data/det3.npy', sess)
            
        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
#         rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
#         onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
        
        
ref = io.loadmat('pnet_dbg.mat')

img_x = np.expand_dims(ref['im_data'], 0)
img_y = np.transpose(img_x, (0,2,1,3))
out = pnet_fun(img_y)
out0 = np.transpose(out[0], (0,2,1,3))
out1 = np.transpose(out[1], (0,2,1,3))

#np.where(abs(out0[0,:,:,:]-ref['out0'])>1e-18)
qqq3 = np.where(abs(out1[0,:,:,:]-ref['out1'])>1e-7) # 3390 diffs with softmax2
print(qqq3[0].shape)
      
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
       
# prob1=sess1.run('prob1:0', feed_dict={data:img})
# print(prob1[0,0,0,:])
# conv42=sess1.run('conv4-2/BiasAdd:0', feed_dict={data:img})
# print(conv42[0,0,0,:])
      
# conv42, prob1 = pnet_fun(img)
# print(prob1[0,0,0,:])
# print(conv42[0,0,0,:])


# [ 0.9929  0.0071] prob1, caffe
# [ 0.9929  0.0071] prob1, tensorflow
  
# [ 0.1207 -0.0116 -0.1231 -0.0463] conv4-2, caffe
# [ 0.1207 -0.0116 -0.1231 -0.0463] conv4-2, tensorflow
      

# g2 = tf.Graph()
# with g2.as_default():
#     data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
#     rnet = align.detect_face.RNet({'data':data})
#     sess2 = tf.Session(graph=g2)
#     rnet.load('../../data/det2.npy', sess2)
#     rnet_fun = lambda img : sess2.run(('conv5-2/conv5-2:0', 'prob1:0'), feed_dict={'input:0':img})
# np.random.seed(666)
# img = np.random.rand(73,3,24,24)
# img = np.transpose(img, (0,2,3,1))
 
# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
#  
# prob1=sess2.run('prob1:0', feed_dict={data:img})
# print(prob1[0,:])
# 
# conv52=sess2.run('conv5-2/conv5-2:0', feed_dict={data:img})
# print(conv52[0,:])
  
# [ 0.9945  0.0055] prob1, caffe
# [ 0.1108 -0.0038 -0.1631 -0.0890] conv5-2, caffe
 
# [ 0.9945  0.0055] prob1, tensorflow
# [ 0.1108 -0.0038 -0.1631 -0.0890] conv5-2, tensorflow

    
# g3 = tf.Graph()
# with g3.as_default():
#     data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
#     onet = align.detect_face.ONet({'data':data})
#     sess3 = tf.Session(graph=g3)
#     onet.load('../../data/det3.npy', sess3)
#     onet_fun = lambda img : sess3.run(('conv6-2/conv6-2:0', 'conv6-3/conv6-3:0', 'prob1:0'), feed_dict={'input:0':img})
# np.random.seed(666)
# img = np.random.rand(11,3,48,48)
# img = np.transpose(img, (0,2,3,1))
 
# np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
#  
# prob1=sess3.run('prob1:0', feed_dict={data:img})
# print(prob1[0,:])
# print('prob1, tensorflow')
# 
# conv62=sess3.run('conv6-2/conv6-2:0', feed_dict={data:img})
# print(conv62[0,:])
# print('conv6-2, tensorflow')
#  
# conv63=sess3.run('conv6-3/conv6-3:0', feed_dict={data:img})
# print(conv63[0,:])
# print('conv6-3, tensorflow')

# [ 0.9988  0.0012] prob1, caffe
# [ 0.0446 -0.0968 -0.1091 -0.0212] conv6-2, caffe
# [ 0.2429  0.6104  0.4074  0.3104  0.5939  0.2729  0.2132  0.5462  0.7863  0.7568] conv6-3, caffe
  
# [ 0.9988  0.0012] prob1, tensorflow
# [ 0.0446 -0.0968 -0.1091 -0.0212] conv6-2, tensorflow
# [ 0.2429  0.6104  0.4074  0.3104  0.5939  0.2729  0.2132  0.5462  0.7863  0.7568] conv6-3, tensorflow

#pnet_fun = lambda img : sess1.run(('conv4-2/BiasAdd:0', 'prob1:0'), feed_dict={'input:0':img})

