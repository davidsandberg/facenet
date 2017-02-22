"""Functions for building the face recognition network.
"""
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

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


def conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, name, phase_train=True, use_batch_norm=True, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        kernel = tf.get_variable("weights", [kH, kW, nIn, nOut],
            initializer=tf.truncated_normal_initializer(stddev=1e-1),
            regularizer=l2_regularizer, dtype=inpOp.dtype)
        cnv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        
        if use_batch_norm:
            conv_bn = batch_norm(cnv, phase_train)
        else:
            conv_bn = cnv
        biases = tf.get_variable("biases", [nOut], initializer=tf.constant_initializer(), dtype=inpOp.dtype)
        bias = tf.nn.bias_add(conv_bn, biases)
        conv1 = tf.nn.relu(bias)
    return conv1

def affine(inpOp, nIn, nOut, name, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        weights = tf.get_variable("weights", [nIn, nOut],
            initializer=tf.truncated_normal_initializer(stddev=1e-1),
            regularizer=l2_regularizer, dtype=inpOp.dtype)
        biases = tf.get_variable("biases", [nOut], initializer=tf.constant_initializer(), dtype=inpOp.dtype)
        affine1 = tf.nn.relu_layer(inpOp, weights, biases)
    return affine1

def l2_loss(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.
    Args:
      tensor: tensor to regularize.
      weight: an optional weight to modulate the loss.
      scope: Optional scope for op_scope.
    Returns:
      the L2 loss op.
    """
    with tf.name_scope(scope):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
    return loss

def lppool(inpOp, pnorm, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        if pnorm == 2:
            pwr = tf.square(inpOp)
        else:
            pwr = tf.pow(inpOp, pnorm)
          
        subsamp = tf.nn.avg_pool(pwr,
                              ksize=[1, kH, kW, 1],
                              strides=[1, dH, dW, 1],
                              padding=padding)
        subsamp_sum = tf.multiply(subsamp, kH*kW)
        
        if pnorm == 2:
            out = tf.sqrt(subsamp_sum)
        else:
            out = tf.pow(subsamp_sum, 1/pnorm)
    
    return out

def mpool(inpOp, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(inpOp,
                       ksize=[1, kH, kW, 1],
                       strides=[1, dH, dW, 1],
                       padding=padding)  
    return maxpool

def apool(inpOp, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        avgpool = tf.nn.avg_pool(inpOp,
                              ksize=[1, kH, kW, 1],
                              strides=[1, dH, dW, 1],
                              padding=padding)
    return avgpool

def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name+'/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name+'/gamma', trainable=True, dtype=x.dtype)
      
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def inception(inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, 
              phase_train=True, use_batch_norm=True, weight_decay=0.0):
  
    print('name = ', name)
    print('inputSize = ', inSize)
    print('kernelSize = {3,5}')
    print('kernelStride = {%d,%d}' % (ks,ks))
    print('outputSize = {%d,%d}' % (o2s2,o3s2))
    print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
    print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
    if (o4s2>0):
        o4 = o4s2
    else:
        o4 = inSize
    print('outputSize = ', o1s+o2s2+o3s2+o4)
    print()
    
    net = []
    
    with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
            if o1s>0:
                conv1 = conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv1)
      
        with tf.variable_scope('branch2_3x3'):
            if o2s1>0:
                conv3a = conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv3 = conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', 'conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv3)
      
        with tf.variable_scope('branch3_5x5'):
            if o3s1>0:
                conv5a = conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv5 = conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', 'conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv5)
      
        with tf.variable_scope('branch4_pool'):
            if poolType=='MAX':
                pool = mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            elif poolType=='L2':
                pool = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)
            
            if o4s2>0:
                pool_conv = conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
            else:
                pool_conv = pool
            net.append(pool_conv)
      
        incept = array_ops.concat(net, 3, name=name)
    return incept
