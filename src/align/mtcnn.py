from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import src.align.detect_face
from scipy import misc

with tf.Graph().as_default():
  
    sess = tf.Session()
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = src.align.detect_face.PNet({'data':data})
            pnet.load('../../data/det1.npy', sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = src.align.detect_face.RNet({'data':data})
            rnet.load('../../data/det2.npy', sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = src.align.detect_face.ONet({'data':data})
            onet.load('../../data/det3.npy', sess)
            
        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

source_path = '/home/david/datasets/casia/CASIA-maxpy-clean/0000045/002.jpg'
img = misc.imread(source_path)

bounding_boxes, points = src.align.detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)

print('Bounding box: %s' % bounding_boxes)


