# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models.network as network

def inference(images, output_dims, keep_probability, phase_train=True, weight_decay=0.0):
    """ Define an inference network for face recognition based 
           on inception modules using batch normalization
    
    Args:
      images: The images to run inference on, dimensions batch_size x height x width x channels
      phase_train: True if batch normalization should operate in training mode
    """
    conv1 = network.conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    pool1 = network.mpool(conv1,  3, 3, 2, 2, 'SAME', 'pool1')
    conv2 = network.conv(pool1,  64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    conv3 = network.conv(conv2,  64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    pool3 = network.mpool(conv3,  3, 3, 2, 2, 'SAME', 'pool3')
  
    incept3a = network.inception(pool3,    192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept3b = network.inception(incept3a, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'MAX', 'incept3b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept3c = network.inception(incept3b, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    
    incept4a = network.inception(incept3c, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'MAX', 'incept4a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept4b = network.inception(incept4a, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'MAX', 'incept4b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept4c = network.inception(incept4b, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'MAX', 'incept4c', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept4d = network.inception(incept4c, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'MAX', 'incept4d', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept4e = network.inception(incept4d, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train, use_batch_norm=True)
    
    incept5a = network.inception(incept4e,    1024, 1, 384, 192, 384, 0, 0, 3, 128, 1, 'MAX', 'incept5a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    incept5b = network.inception(incept5a, 896, 1, 384, 192, 384, 0, 0, 3, 128, 1, 'MAX', 'incept5b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    pool6 = network.apool(incept5b,  3, 3, 1, 1, 'VALID', 'pool6')
    resh1 = tf.reshape(pool6, [-1, 896])
    
    fc7a = network.affine(resh1, 896, output_dims[0], 'fc7a', weight_decay=0.0)
    logits1 = tf.nn.dropout(fc7a, keep_probability)
    
    fc7b = network.affine(resh1, 896, output_dims[1], 'fc7b', weight_decay=weight_decay)
    logits2 = tf.nn.dropout(fc7b, keep_probability)
  
    return logits1, logits2
