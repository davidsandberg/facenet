"""Functions for building the face recognition network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import glob
from os import path


import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import gfile
import numpy as np
from scipy import misc

FLAGS = tf.app.flags.FLAGS

parameters = []

conv_counter = 1
pool_counter = 1
affine_counter = 1

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, prefix, phase_train=True, use_batch_norm=True):
  global conv_counter
  global parameters
  name = prefix + '_' + str(conv_counter)
  conv_counter += 1
  with tf.name_scope(name) as scope:
    kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
    
    if use_batch_norm:
      conv_bn = _batch_norm(conv, nOut, phase_train, name+'_bn')
    else:
      conv_bn = conv
    biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.reshape(tf.nn.bias_add(conv_bn, biases), conv.get_shape())
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
  
def _lppool(inpOp, pnorm, kH, kW, dH, dW, padding):
  global pool_counter
  global parameters
  name = 'pool' + str(pool_counter)
  pool_counter += 1
  
  with tf.name_scope('lppool') as scope:
    if pnorm == 2:
      pwr = tf.square(inpOp)
    else:
      pwr = tf.pow(inpOp, pnorm)
      
    subsamp = tf.nn.avg_pool(pwr,
                          ksize=[1, kH, kW, 1],
                          strides=[1, dH, dW, 1],
                          padding=padding,
                          name=name)
    subsamp_sum = tf.mul(subsamp, kH*kW)
    
    if pnorm == 2:
      out = tf.sqrt(subsamp_sum)
    else:
      out = tf.pow(subsamp_sum, 1/pnorm)
    
  return out

def _mpool(inpOp, kH, kW, dH, dW, padding):
  global pool_counter
  global parameters
  name = 'pool' + str(pool_counter)
  pool_counter += 1
  with tf.name_scope('maxpool') as scope:
    maxpool = tf.nn.max_pool(inpOp,
                   ksize=[1, kH, kW, 1],
                   strides=[1, dH, dW, 1],
                   padding=padding,
                   name=name)  
  return maxpool

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

def _batch_norm(x, n_out, phase_train, name, scope='bn', affine=True):
  """
  Batch normalization on convolutional maps.
  Args:
      x:           Tensor, 4D BHWD input maps
      n_out:       integer, depth of input maps
      phase_train: boolean tf.Variable, true indicates training phase
      scope:       string, variable scope
      affine:      whether to affine-transform outputs
  Return:
      normed:      batch-normalized maps
  Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
  """
  global parameters

  beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                     name=name+'/beta', trainable=True)
  gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                      name=name+'/gamma', trainable=affine)

  batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
  ema = tf.train.ExponentialMovingAverage(decay=0.9)
  ema_apply_op = ema.apply([batch_mean, batch_var])
  ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
  def mean_var_with_update():
    with tf.control_dependencies([ema_apply_op]):
      return tf.identity(batch_mean), tf.identity(batch_var)
  mean, var = control_flow_ops.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema_mean, ema_var))
  normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                      beta, gamma, 1e-3, affine, name=name)
  parameters += [beta, gamma]
  return normed

def _inception(inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, phase_train=True, use_batch_norm=True):
  
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
  
  if o1s>0:
    conv1 = _conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', name+'_in1_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
    net.append(conv1)

  if o2s1>0:
    conv3a = _conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', name+'_in2_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
    conv3 = _conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', name+'_in2_conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm)
    net.append(conv3)

  if o3s1>0:
    conv5a = _conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', name+'_in3_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
    conv5 = _conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', name+'_in3_conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm)
    net.append(conv5)

  if poolType=='max':
    pool = _mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME')
  elif poolType=='l2':
    pool = _lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME')
  else:
    raise ValueError('Invalid pooling type "%s"' % poolType)
  
  if o4s2>0:
    pool_conv = _conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', name+'_in4_conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm)
  else:
    pool_conv = pool
  net.append(pool_conv)

  incept = array_ops.concat(3, net, name=name)
  return incept

def inference_nn4_max_pool_96(images, phase_train=True):
  """ Define an inference network for face recognition based 
         on inception modules using batch normalization
  
  Args:
    images: The images to run inference on, dimensions batch_size x height x width x channels
    phase_train: True if batch normalization should operate in training mode
  """
  conv1 = _conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train, use_batch_norm=True)
  pool1 = _mpool(conv1,  3, 3, 2, 2, 'SAME')
  conv2 = _conv(pool1,  64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True)
  conv3 = _conv(conv2,  64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True)
  pool3 = _mpool(conv3,  3, 3, 2, 2, 'SAME')

  incept3a = _inception(pool3,      192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'max', 'incept3a', phase_train=phase_train, use_batch_norm=True)
  incept3b = _inception(incept3a, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'max', 'incept3b', phase_train=phase_train, use_batch_norm=True)
  incept3c = _inception(incept3b, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'max', 'incept3c', phase_train=phase_train, use_batch_norm=True)
  
  incept4a = _inception(incept3c, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'max', 'incept4a', phase_train=phase_train, use_batch_norm=True)
  incept4b = _inception(incept4a, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'max', 'incept4b', phase_train=phase_train, use_batch_norm=True)
  incept4c = _inception(incept4b, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'max', 'incept4c', phase_train=phase_train, use_batch_norm=True)
  incept4d = _inception(incept4c, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'max', 'incept4d', phase_train=phase_train, use_batch_norm=True)
  incept4e = _inception(incept4d, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'max', 'incept4e', phase_train=phase_train, use_batch_norm=True)
  
  incept5a = _inception(incept4e,    1024, 1, 384, 192, 384, 0, 0, 3, 128, 1, 'max', 'incept5a', phase_train=phase_train, use_batch_norm=True)
  incept5b = _inception(incept5a, 896, 1, 384, 192, 384, 0, 0, 3, 128, 1, 'max', 'incept5b', phase_train=phase_train, use_batch_norm=True)
  pool6 = _apool(incept5b,  3, 3, 1, 1, 'VALID')

  resh1 = tf.reshape(pool6, [-1, 896])
  affn1 = _affine(resh1, 896, 128)
  norm = tf.nn.l2_normalize(affn1, 1, 1e-10)

  return norm

def inference_vggface_96(images, phase_train=True):
    #(1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
    conv1 = _conv(images, 3, 64, 3, 3, 1, 1, 'SAME', 'conv1_3x3', phase_train=phase_train, use_batch_norm=False)
    #(3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
    conv2 = _conv(conv1, 64, 64, 3, 3, 1, 1, 'SAME', 'conv2_3x3', phase_train=phase_train, use_batch_norm=False)
    #(5): nn.SpatialMaxPooling(2,2,2,2)
    pool1 = _mpool(conv2,  2, 2, 2, 2, 'SAME')
    #(6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
    conv3 = _conv(pool1, 64, 128, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=False)
    #(8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
    conv4 = _conv(conv3, 128, 128, 3, 3, 1, 1, 'SAME', 'conv4_3x3', phase_train=phase_train, use_batch_norm=False)
    #(10): nn.SpatialMaxPooling(2,2,2,2)
    pool2 = _mpool(conv4,  2, 2, 2, 2, 'SAME')
    #(11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
    conv5 = _conv(pool2, 128, 256, 3, 3, 1, 1, 'SAME', 'conv5_3x3', phase_train=phase_train, use_batch_norm=False)
    #(13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
    conv6 = _conv(conv5, 256, 256, 3, 3, 1, 1, 'SAME', 'conv6_3x3', phase_train=phase_train, use_batch_norm=False)
    #(15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
    conv7 = _conv(conv6, 256, 256, 3, 3, 1, 1, 'SAME', 'conv7_3x3', phase_train=phase_train, use_batch_norm=False)
    #(17): nn.SpatialMaxPooling(2,2,2,2)
    pool3 = _mpool(conv7,  2, 2, 2, 2, 'SAME')
    #(18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
    conv8 = _conv(pool3, 256, 512, 3, 3, 1, 1, 'SAME', 'conv8_3x3', phase_train=phase_train, use_batch_norm=False)
    #(20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
    conv9 = _conv(conv8, 512, 512, 3, 3, 1, 1, 'SAME', 'conv9_3x3', phase_train=phase_train, use_batch_norm=False)
    #(22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
    conv10 = _conv(conv9, 512, 512, 3, 3, 1, 1, 'SAME', 'conv10_3x3', phase_train=phase_train, use_batch_norm=False)
    #(24): nn.SpatialMaxPooling(2,2,2,2)
    pool4 = _mpool(conv10,  2, 2, 2, 2, 'SAME')
    #(25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
    conv11 = _conv(pool4, 512, 512, 3, 3, 1, 1, 'SAME', 'conv11_3x3', phase_train=phase_train, use_batch_norm=False)
    #(27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
    conv12 = _conv(conv11, 512, 512, 3, 3, 1, 1, 'SAME', 'conv12_3x3', phase_train=phase_train, use_batch_norm=False)
    #(29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
    conv13 = _conv(conv12, 512, 512, 3, 3, 1, 1, 'SAME', 'conv13_3x3', phase_train=phase_train, use_batch_norm=False)
    #(31): nn.SpatialMaxPooling(2,2,2,2)
    pool5 = _mpool(conv13,  2, 2, 2, 2, 'SAME')
    #(32): nn.View
    resh1 = tf.reshape(pool5, [-1, 4608])
    #(33): nn.Linear(25088 -> 4096)
    #affn1 = _affine(resh1, 4608, 4096)
    #(35): nn.Dropout(0.500000)
    #(36): nn.Linear(4096 -> 4096)
    #affn2 = _affine(affn1, 4096, 4096)
    #(38): nn.Dropout(0.500000)
    #(39): nn.Linear(4096 -> 2622)
    affn2 = _affine(resh1, 4608, 128)
    norm = tf.nn.l2_normalize(affn2, 1, 1e-4)
    
    return norm

def triplet_loss(anchor, positive, negative):
  """Calculate the triplet loss according to the FaceNet paper
  
  Args:
    anchor: the embeddings for the anchor images.
    positive: the embeddings for the positive images.
    positive: the embeddings for the negative images.

  Returns:
    the triplet loss according to the FaceNet paper as a float tensor.
  """
  with tf.name_scope('triplet_loss') as scope:
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)  # Summing over distances in each batch
    neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
    
    basic_loss = tf.add(tf.sub(pos_dist,neg_dist), FLAGS.alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0, name='tripletloss')
    
  return loss

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train(total_loss, global_step):
  """Setup training for the FaceNet model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    grads = opt.compute_gradients(total_loss)
    
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.moving_average_decay, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op, grads

def load_data(image_paths):
  nrof_samples = len(image_paths)
  img_list = [None] * nrof_samples
  for i in xrange(nrof_samples):
    img_list[i] = prewhiten(misc.imread(image_paths[i]))
  images = np.stack(img_list)
  return images

def prewhiten(x):
  mean = np.mean(x)
  std = np.std(x)
  std_adj = np.max(std, 1.0/np.sqrt(x.size))
  y = np.multiply(np.subtract(x, mean), 1/std_adj)
  return y  

def get_batch(image_data, batch_size, batch_index):
  nrof_examples = np.size(image_data, 0)
  j = batch_index*batch_size % nrof_examples
  if j+batch_size<=nrof_examples:
    batch = image_data[j:j+batch_size,:,:,:]
  else:
    x1 = image_data[j:nrof_examples,:,:,:]
    x2 = image_data[0:nrof_examples-j,:,:,:]
    batch = np.vstack([x1,x2])
  batch_float = batch.astype(np.float32)
  return batch_float

def get_triplet_batch(triplets, batch_index):
  ax, px, nx = triplets
  a = get_batch(ax, int(FLAGS.batch_size/3), batch_index)
  p = get_batch(px, int(FLAGS.batch_size/3), batch_index)
  n = get_batch(nx, int(FLAGS.batch_size/3), batch_index)
  batch = np.vstack([a, p, n])
  return batch

def select_training_triplets(embeddings, num_per_class, image_data):

  def dist(emb1, emb2):
    x = np.square(np.subtract(emb1, emb2))
    return np.sum(x, 0)

  nrof_images = image_data.shape[0]
  nrof_triplets = nrof_images - FLAGS.people_per_batch
  shp = [nrof_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
  as_arr = np.zeros(shp)
  ps_arr = np.zeros(shp)
  ns_arr = np.zeros(shp)
  
  trip_idx = 0
  shuffle = np.arange(nrof_triplets)
  np.random.shuffle(shuffle)
  emb_start_idx = 0
  nrof_random_negs = 0
  for i in xrange(FLAGS.people_per_batch):
    n = num_per_class[i]
    for j in range(1,n):
      a_idx = emb_start_idx
      p_idx = emb_start_idx + j
      as_arr[shuffle[trip_idx]] = image_data[a_idx]
      ps_arr[shuffle[trip_idx]] = image_data[p_idx]

      # Select a semi-hard negative that has a distance
      #  further away from the positive exemplar.
      pos_dist = dist(embeddings[a_idx][:], embeddings[p_idx][:])
      sel_neg_idx = emb_start_idx
      while sel_neg_idx>=emb_start_idx and sel_neg_idx<=emb_start_idx+n-1:
        sel_neg_idx = (np.random.randint(1, 2**32) % nrof_images) -1  # Seems to give the same result as the lua implementation
        #sel_neg_idx = np.random.random_integers(0, nrof_images-1)
      sel_neg_dist = dist(embeddings[a_idx][:], embeddings[sel_neg_idx][:])

      random_neg = True
      for k in range(nrof_images):
        if k<emb_start_idx or k>emb_start_idx+n-1:
          neg_dist = dist(embeddings[a_idx][:], embeddings[k][:])
          if pos_dist<neg_dist and neg_dist<sel_neg_dist and np.abs(pos_dist-neg_dist)<FLAGS.alpha:
            random_neg = False
            sel_neg_dist = neg_dist
            sel_neg_idx = k
      
      if random_neg:
        nrof_random_negs += 1
        
      ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
      #print('Triplet %d: (%d, %d, %d), pos_dist=%2.3f, neg_dist=%2.3f, sel_neg_dist=%2.3f' % (trip_idx, a_idx, p_idx, sel_neg_idx, pos_dist, neg_dist, sel_neg_dist))
      trip_idx += 1
      
    emb_start_idx += n
  
  triplets = (as_arr, ps_arr, ns_arr)
  
  return triplets, nrof_random_negs, nrof_triplets

  
def select_validation_triplets(num_per_class, people_per_batch, image_data):

  nrof_images = image_data.shape[0]
  nrof_triplets = nrof_images - people_per_batch
  shp = [nrof_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
  as_arr = np.zeros(shp)
  ps_arr = np.zeros(shp)
  ns_arr = np.zeros(shp)
  
  trip_idx = 0
  shuffle = np.arange(nrof_triplets)
  np.random.shuffle(shuffle)
  emb_start_idx = 0
  nrof_random_negs = 0
  for i in xrange(people_per_batch):
    n = num_per_class[i]
    for j in range(1,n):
      a_idx = emb_start_idx
      p_idx = emb_start_idx + j
      as_arr[shuffle[trip_idx]] = image_data[a_idx]
      ps_arr[shuffle[trip_idx]] = image_data[p_idx]

      # Select a random negative example
      sel_neg_idx = emb_start_idx
      while sel_neg_idx>=emb_start_idx and sel_neg_idx<=emb_start_idx+n-1:
        sel_neg_idx = (np.random.randint(1, 2**32) % nrof_images) -1

      ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
      #print('Triplet %d: (%d, %d, %d), pos_dist=%2.3f, neg_dist=%2.3f, sel_neg_dist=%2.3f' % (trip_idx, a_idx, p_idx, sel_neg_idx, pos_dist, neg_dist, sel_neg_dist))
      trip_idx += 1
      
    emb_start_idx += n
  
  triplets = (as_arr, ps_arr, ns_arr)
  
  return triplets, nrof_triplets
  

class ImageClass():
  "Stores the paths to images for a given class"
  def __init__(self, name, image_paths):
    self.name = name
    self.image_paths = image_paths

  def __str__(self):
    return self.name + ', ' + str(len(self.image_paths)) + ' images'

  def __len__(self):
    return len(self.image_paths)
  
def get_dataset(paths):
  dataset = []
  for path in paths.split(':'):
    classes = os.listdir(path)
    classes.sort()
    nrof_classes = len(classes)
    #dataset = [None] * nrof_classes
    for i in range(nrof_classes):
      class_name = classes[i]
      facedir = os.path.join(path, class_name)
      images = os.listdir(facedir)
      image_paths = map(lambda x: os.path.join(facedir,x), images)
      dataset.append(ImageClass(class_name, image_paths))

  return dataset

def split_dataset(dataset, split_ratio):
  nrof_classes = len(dataset)
  class_indices = np.arange(nrof_classes)
  np.random.shuffle(class_indices)
  split = int(round(nrof_classes*split_ratio))
  train_set = [dataset[i] for i in class_indices[0:split]]  # Range does not include the last index
  test_set = [dataset[i] for i in class_indices[split:nrof_classes]]
  return train_set, test_set

def sample_people(dataset, people_per_batch, images_per_person):
  nrof_images = people_per_batch * images_per_person

  # Sample classes from the dataset
  nrof_classes = len(dataset)
  class_indices = np.arange(nrof_classes)
  np.random.shuffle(class_indices)
  
  i = 0
  image_paths = []
  num_per_class = []
  # Sample images from these classes until we have enough
  while len(image_paths)<nrof_images:
    class_index = class_indices[i]
    nrof_images_in_class = len(dataset[class_index])
    image_indices = np.arange(nrof_images_in_class)
    np.random.shuffle(image_indices)
    nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
    idx = image_indices[0:nrof_images_from_class]
    image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
    image_paths += image_paths_for_class
    num_per_class.append(nrof_images_from_class)
    i+=1

  return image_paths, num_per_class

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame):
  tpr_array = []
  fpr_array = []
  accuracy_array = []
  predict_issame = [None] * len(actual_issame)
  predict_issame_stored = []
  dist_stored = []
  for threshold in thresholds:
    tp = tn = fp = fn = 0
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)

    for i in range(dist.size):
      predict_issame[i] = dist[i] < threshold
      if predict_issame[i] and actual_issame[i]:
        tp += 1
      elif predict_issame[i] and not actual_issame[i]:
        fp += 1
      elif not predict_issame[i] and not actual_issame[i]:
        tn += 1
      elif not predict_issame[i] and actual_issame[i]:
        fn += 1

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    accuracy = float(tp+tn)/dist.size
    
    if len(accuracy_array)>0 and accuracy>=np.max(accuracy_array):
      predict_issame_stored = predict_issame
      dist_stored = dist

    tpr_array.append(tpr)
    fpr_array.append(fpr)
    accuracy_array.append(accuracy)

  return tpr_array, fpr_array, accuracy_array, predict_issame_stored, dist_stored
