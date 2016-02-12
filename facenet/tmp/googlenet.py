from datetime import datetime
import math
import time

import tensorflow.python.platform
import tensorflow as tf
from tensorflow.python.ops import array_ops

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 80,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1

def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def _mpool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.nn.max_pool(inpOp,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding=padding,
                          name=name)

def _apool(inpOp, kH, kW, dH, dW, padding):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.nn.avg_pool(inpOp,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding=padding,
                          name=name)

def _inception(inp, inSize, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2):
    conv1 = _conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME')

    conv3_ = _conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME')
    conv3 = _conv(conv3_, o2s1, o2s2, 3, 3, 1, 1, 'SAME')

    conv5_ = _conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME')
    conv5 = _conv(conv5_, o3s1, o3s2, 5, 5, 1, 1, 'SAME')

    pool_ = _mpool(inp, o4s1, o4s1, 1, 1, 'SAME')
    pool = _conv(pool_, inSize, o4s2, 1, 1, 1, 1, 'SAME')

    incept = array_ops.concat(3, [conv1, conv3, conv5, pool])
    return incept


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def inference(images):
    conv1 = _conv (images, 3, 64, 7, 7, 2, 2, 'SAME')
    pool1 = _mpool(conv1,  3, 3, 2, 2, 'SAME')
    conv2 = _conv (pool1,  64, 64, 1, 1, 1, 1, 'SAME')
    conv3 = _conv (conv2,  64, 192, 3, 3, 1, 1, 'SAME')
    pool3 = _mpool(conv3,  3, 3, 2, 2, 'SAME')

    incept3a = _inception(pool3,    192, 64, 96, 128, 16, 32, 3, 32)
    incept3b = _inception(incept3a, 256, 128, 128, 192, 32, 96, 3, 64)
    pool4 = _mpool(incept3b,  3, 3, 2, 2, 'SAME')
    incept4a = _inception(pool4,    480, 192,  96, 208, 16, 48, 3, 64)
    incept4b = _inception(incept4a, 512, 160, 112, 224, 24, 64, 3, 64)
    incept4c = _inception(incept4b, 512, 128, 128, 256, 24, 64, 3, 64)
    incept4d = _inception(incept4c, 512, 112, 144, 288, 32, 64, 3, 64)
    incept4e = _inception(incept4d, 528, 256, 160, 320, 32, 128, 3, 128)
    pool5 = _mpool(incept4e,  3, 3, 2, 2, 'SAME')
    incept5a = _inception(pool5,    832, 256, 160, 320, 32, 128, 3, 128)
    incept5b = _inception(incept5a, 832, 384, 192, 384, 48, 128, 3, 128)
    pool6 = _apool(incept5b,  7, 7, 1, 1, 'VALID')

    resh1 = tf.reshape(pool6, [-1, 1024])
    affn1 = _affine(resh1, 1024, 1000)

    return affn1

