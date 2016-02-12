# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform  # @UnusedImport
from tensorflow.python.platform import gfile

from six.moves import xrange  # pylint: disable=redefined-builtin @UnresolvedImport
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py    # HDF5 support
import facenet

FLAGS = tf.app.flags.FLAGS

time_str = datetime.strftime(datetime.now(), '/%Y%m%d-%H%M%S')

tf.app.flags.DEFINE_string('train_dir', '/home/david/logs/openface' + time_str,
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_nrof_epochs', 200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 90,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/david/datasets/fs_aligned/',
                           """Path to the data directory containing aligned face patches.""")
tf.app.flags.DEFINE_integer('people_per_batch', 45,
                            """Number of people per batch.""")
tf.app.flags.DEFINE_integer('images_per_person', 40,
                            """Number of images per person.""")
tf.app.flags.DEFINE_integer('epoch_size', 1000,
                            """Number of mini batches per epoch.""")
tf.app.flags.DEFINE_integer('alpha', 0.2,
                            """Positive to negative triplet distance margin.""")
tf.app.flags.DEFINE_integer('initial_learning_rate', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('learning_rate_decay_factor', 0.1,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 50,
                            """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_integer('moving_average_decay', 0.9999,
                            """Epochs after which learning rate decays.""")

conv_counter = 1
pool_counter = 1
affine_counter = 1


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, prefix, phase_train=True, use_batch_norm=True, init_weight=None, init_bias=None):
  global conv_counter
  global parameters
  name = prefix + '_' + str(conv_counter)
  conv_counter += 1
  with tf.name_scope(name) as scope:
    if init_weight is None:
      init_weight = tf.truncated_normal([kH, kW, nIn, nOut], dtype=tf.float32, stddev=1e-1)
    if init_bias is None:
      init_bias = tf.constant(0.0, shape=[nOut], dtype=tf.float32)
    kernel = tf.Variable(init_weight, name='weights')
    conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
    
    if use_batch_norm:
      conv_bn = batch_norm(conv, nOut, phase_train, name+'_bn', False)
    else:
      conv_bn = conv
    biases = tf.Variable(init_bias, trainable=True, name='biases')
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

def train():
  dataset = facenet.get_dataset(FLAGS.data_dir)
  train_set, test_set = facenet.split_dataset(dataset, 0.9)
  
  fileName = "/home/david/debug4.h5"
  f = h5py.File(fileName,  "r")
  for item in f.values():
    print(item)
  
  w1 = f['1w'][:]
  b1 = f['1b'][:]
  f.close()
  print(w1.shape)
  print(b1.shape)
  
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Placeholder for input images
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 96, 96, 3), name='Input')
    
    # Build a Graph that computes the logits predictions from the inference model
    #embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=True)
    
    conv1 = _conv(images_placeholder, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=False, use_batch_norm=False, init_weight=w1, init_bias=b1)
    resh1 = tf.reshape(conv1, [-1, 294912])
    embeddings = _affine(resh1, 294912, 128)
    
        
    # Split example embeddings into anchor, positive and negative
    a, p, n = tf.split(0, 3, embeddings)

    # Calculate triplet loss
    loss = facenet.triplet_loss(a, p, n)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_op, grads = facenet.train(loss, global_step)
    
    # Create a saver
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
    
    epoch = 0
    
    with sess.as_default():

      while epoch<FLAGS.max_nrof_epochs:
        batch_number = 0
        while batch_number<FLAGS.epoch_size:
          print('Loading new data')
          image_data, num_per_class, image_paths = facenet.load_data(train_set)
      
          print('Selecting suitable triplets for training')
          start_time = time.time()
          emb_list = []
          # Run a forward pass for the sampled images
          nrof_examples_per_epoch = FLAGS.people_per_batch*FLAGS.images_per_person
          nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch/FLAGS.batch_size))
          if True:
            for i in xrange(nrof_batches_per_epoch):
              feed_dict, _ = facenet.get_batch(images_placeholder, image_data, i)
              emb_list += sess.run([embeddings], feed_dict=feed_dict)
            emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
            # Select triplets based on the embeddings
            apn, nrof_random_negs, nrof_triplets = facenet.select_triplets(emb_array, num_per_class, image_data)
            duration = time.time() - start_time
            print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (nrof_random_negs, nrof_triplets, duration))
            
            count = 0
            while count<nrof_triplets*3 and batch_number<FLAGS.epoch_size:
              start_time = time.time()
              feed_dict, batch = facenet.get_batch(images_placeholder, apn, batch_number)
              if (batch_number%20==0):
                err, summary_str, _  = sess.run([loss, summary_op, train_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, FLAGS.epoch_size*epoch+batch_number)
              else:
                err, _  = sess.run([loss, train_op], feed_dict=feed_dict)
              duration = time.time() - start_time
              print('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %2.3f' % (epoch, batch_number, FLAGS.epoch_size, duration, err))
              batch_number+=1
              count+=FLAGS.batch_size

          else:
  
            while batch_number<FLAGS.epoch_size:
              start_time = time.time()
              feed_dict, _ = facenet.get_batch(images_placeholder, image_data, batch_number)
              
              grad_tensors, grad_vars = zip(*grads)
              eval_list = (train_op, loss) + grad_tensors
              result  = sess.run(eval_list, feed_dict=feed_dict)
              grads_eval = result[2:]
              nrof_parameters = 0
              for gt, gv in zip(grads_eval, grad_vars):
                print('%40s: %6d' % (gv.op.name, np.size(gt)))
                nrof_parameters += np.size(gt)
              print('Total number of parameters: %d' % nrof_parameters)
              err = result[1]
              batch_number+=1
        epoch+=1

      # Save the model checkpoint periodically.
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=epoch*FLAGS.epoch_size+batch_number)

def plot_triplet(apn, idx):
  plt.subplot(1,3,1)
  plt.imshow(np.multiply(apn[idx*3+0,:,:,:],1/256))
  plt.subplot(1,3,2)
  plt.imshow(np.multiply(apn[idx*3+1,:,:,:],1/256))
  plt.subplot(1,3,3)
  plt.imshow(np.multiply(apn[idx*3+2,:,:,:],1/256))
  
def main(argv=None):  # pylint: disable=unused-argument
  if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
  gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
