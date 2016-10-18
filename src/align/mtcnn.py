from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import src.align.detect_face
from scipy import misc

g1 = tf.Graph()
with g1.as_default():
    with tf.name_scope('pnet') as scope:
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = src.align.detect_face.PNet({'data':data})
        sess1 = tf.Session(graph=g1)
        pnet.load('../../data/det1.npy', sess1)
    pnet_fun = lambda img : sess1.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})

g2 = tf.Graph()
with g2.as_default():
    data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
    rnet = src.align.detect_face.RNet({'data':data})
    sess2 = tf.Session(graph=g2)
    rnet.load('../../data/det2.npy', sess2)
    rnet_fun = lambda img : sess2.run(('conv5-2/conv5-2:0', 'prob1:0'), feed_dict={'input:0':img})
 
g3 = tf.Graph()
with g3.as_default():
    data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
    onet = src.align.detect_face.ONet({'data':data})
    sess3 = tf.Session(graph=g3)
    onet.load('../../data/det3.npy', sess3)
    onet_fun = lambda img : sess3.run(('conv6-2/conv6-2:0', 'conv6-3/conv6-3:0', 'prob1:0'), feed_dict={'input:0':img})

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

source_path = '/home/david/datasets/casia/CASIA-maxpy-clean/0000045/002.jpg'
img = misc.imread(source_path)

bounding_boxes, points = src.align.detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)

print('Bounding box: %s' % bounding_boxes)


