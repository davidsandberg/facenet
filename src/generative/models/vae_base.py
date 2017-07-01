'''
Created on Jul 1, 2017

@author: david
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Vae(object):
  
    def __init__(self, latent_variable_dim):
        self.latent_variable_dim = latent_variable_dim
        self.batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
  
    def encoder(self, images):
        # Note: change relu to leaky relu
        weight_decay = 0.0
        with tf.variable_scope('encoder'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params):
                net = slim.conv2d(images, 32, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_1')
                net = slim.conv2d(net, 64, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_2')
                net = slim.conv2d(net, 128, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_3')
                net = slim.conv2d(net, 256, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_4')
                net = slim.flatten(net)
                fc1 = slim.fully_connected(net, self.latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_1')
                fc2 = slim.fully_connected(net, self.latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_2')
        return fc1, fc2
      
    def decoder(self, latent_var):
        # TODO: Maybe change relu to leaky relu
        weight_decay = 0.0
        with tf.variable_scope('decoder'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params):
                net = slim.fully_connected(latent_var, 4096, activation_fn=None, normalizer_fn=None, scope='Fc_1')
                net = tf.reshape(net, [-1,4,4,256], name='Reshape')
                
                net = tf.image.resize_nearest_neighbor(net, size=(8,8), name='Upsample_1')
                net = slim.conv2d(net, 128, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_1')
        
                net = tf.image.resize_nearest_neighbor(net, size=(16,16), name='Upsample_2')
                net = slim.conv2d(net, 64, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_2')
        
                net = tf.image.resize_nearest_neighbor(net, size=(32,32), name='Upsample_3')
                net = slim.conv2d(net, 32, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_3')
        
                net = tf.image.resize_nearest_neighbor(net, size=(64,64), name='Upsample_4')
                net = slim.conv2d(net, 3, [3, 3], 1, activation_fn=None, scope='Conv2d_4')
                
        return net
      
