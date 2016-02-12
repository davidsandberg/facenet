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
import facenet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/david/logs/openface',
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
tf.app.flags.DEFINE_integer('initial_learning_rate', 0.005,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('learning_rate_decay_factor', 0.1,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 350,
                            """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_integer('moving_average_decay', 0.9999,
                            """Epochs after which learning rate decays.""")


def train():
  dataset = facenet.get_dataset(FLAGS.data_dir)
  train_set, test_set = facenet.split_dataset(dataset, 0.9)
  
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Placeholder for input images
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 96, 96, 3), name='Input')
    
    # Build a Graph that computes the logits predictions from the inference model
    embeddings = facenet.inference_no_batch_norm_deeper(images_placeholder, tf.constant(True))
    #embeddings = facenet.inference(images_placeholder, tf.constant(False))
    
    # Split example embeddings into anchor, positive and negative
    #a, p, n = tf.split(0, 3, embeddings)

    # Calculate triplet loss
    loss = facenet.triplet_loss_modified(embeddings)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_op, grads = facenet.train(loss, global_step)
    
    # Create a saver
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    
    check_num = tf.add_check_numerics_ops()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
    
    epoch = 1
    
    with sess.as_default():

      while epoch<FLAGS.max_nrof_epochs:
        batch_number = 0
        while batch_number<FLAGS.epoch_size:
          print('Loading new data')
          image_data, num_per_class = facenet.load_data(train_set)
      
          print('Selecting suitable triplets for training')
          start_time = time.time()
          emb_list = []
          # Run a forward pass for the sampled images
          nrof_examples_per_epoch = FLAGS.people_per_batch*FLAGS.images_per_person
          nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch/FLAGS.batch_size))
          #for i in xrange(nrof_batches_per_epoch):
            #feed_dict = facenet.get_batch(images_placeholder, image_data, i)
            #emb_list += sess.run([embeddings], feed_dict=feed_dict)
          #emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
          ## Select triplets based on the embeddings
          #apn, nrof_random_negs, nrof_triplets = facenet.select_triplets(emb_array, num_per_class, image_data)
          #duration = time.time() - start_time
          #print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (nrof_random_negs, nrof_triplets, duration))
    
          count = 0
  #        while count<nrof_triplets*3 and batch_number<FLAGS.epoch_size:
          while batch_number<FLAGS.epoch_size:
            start_time = time.time()
   #         feed_dict = facenet.get_batch(images_placeholder, apn, batch_number)
            feed_dict = facenet.get_batch(images_placeholder, image_data, batch_number)
            
            grad_tensors, grad_vars = zip(*grads)
            grads_eval  = sess.run(grad_tensors, feed_dict=feed_dict)
            for gt, gv in zip(grads_eval, grad_vars):
              print('%40s: %6d %6f  %6f' % (gv.op.name, np.sum(np.isnan(gt)), np.max(gt), np.min(gt)))
            
            duration = time.time() - start_time
            
            print('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %2.3f' % (epoch, batch_number, FLAGS.epoch_size, duration, err))
            batch_number+=1
            count+=FLAGS.batch_size
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
