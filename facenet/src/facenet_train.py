"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform  # @UnusedImport
from tensorflow.python.platform import gfile

import tensorflow as tf
import numpy as np
import facenet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('logs_base_dir', '/home/david/logs/facenet',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('models_base_dir', '/home/david/models/facenet',
                           """Directory where to write trained models and checkpoints.""")
#tf.app.flags.DEFINE_string('model_name', '20160228-110932-1',
                           #"""XXXXXXXX.""")
tf.app.flags.DEFINE_string('model_name', '',
                           """XXXXXXXX.""")
tf.app.flags.DEFINE_string('data_dir', '/home/david/datasets/facescrub/fs_aligned:/home/david/datasets/casia/casia-webface-aligned',
                           """Path to the data directory containing aligned face patches.""")
tf.app.flags.DEFINE_integer('max_nrof_epochs', 200,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 90,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_integer('people_per_batch', 45,
                            """Number of people per batch.""")
tf.app.flags.DEFINE_integer('images_per_person', 40,
                            """Number of images per person.""")
tf.app.flags.DEFINE_integer('epoch_size', 20,
                            """Number of batches per epoch.""")
tf.app.flags.DEFINE_float('alpha', 0.2,
                            """Positive to negative triplet distance margin.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """Expontential decay for tracking of training parameters.""")
tf.app.flags.DEFINE_float('train_set_fraction', 0.9,
                          """Fraction of the data set that is used for training.""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """Random seed.""")

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.model_name:
    subdir = FLAGS.model_name
    preload_model = True
  else:
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    preload_model = False
  log_dir = os.path.join(FLAGS.logs_base_dir, subdir)
  model_dir = os.path.join(FLAGS.models_base_dir, subdir)
  if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
    os.mkdir(model_dir)
  
  np.random.seed(seed=FLAGS.seed)
  dataset = facenet.get_dataset(FLAGS.data_dir)
  train_set, test_set = facenet.split_dataset(dataset, FLAGS.train_set_fraction)
  
  with tf.Graph().as_default():
    tf.set_random_seed(FLAGS.seed)
    global_step = tf.Variable(0, trainable=False)

    # Placeholder for input images
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='Input')
    
    # Placeholder for phase_train
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    
    # Build the inference graph
    embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=phase_train_placeholder)
    #embeddings = facenet.inference_vggface_96(images_placeholder)
    
    # Split example embeddings into anchor, positive and negative
    anchor, positive, negative = tf.split(0, 3, embeddings)

    # Calculate triplet loss
    loss = facenet.triplet_loss(anchor, positive, negative)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_op, grads = facenet.train(loss, global_step)
    
    # Create a saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(log_dir, graph_def=sess.graph_def)
    
    epoch = 0
    
    with sess.as_default():

      if preload_model:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          raise ValueError('Checkpoint not found')

      while epoch<FLAGS.max_nrof_epochs:
        # Train for one epoch
        step = train_epoch(sess, train_set, epoch, images_placeholder, phase_train_placeholder,
                    global_step, embeddings, loss, train_op, summary_op, summary_writer)
        # Validate epoch
        validate_epoch(sess, train_set, epoch, images_placeholder, phase_train_placeholder,
                       global_step, embeddings, loss, train_op, summary_op)
        # Save the model checkpoint after each epoch
        print('Saving checkpoint')
        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        graphdef_dir = os.path.join(model_dir, 'graphdef')
        graphdef_filename = 'graph_def.pb'
        if (not os.path.exists(os.path.join(graphdef_dir, graphdef_filename))):
          print('Saving graph definition')
          tf.train.write_graph(sess.graph_def, graphdef_dir, graphdef_filename, False)
        epoch+=1

def train_epoch(sess, dataset, epoch, images_placeholder, phase_train_placeholder, 
                global_step, embeddings, loss, train_op, summary_op, summary_writer):
  batch_number = 0
  while batch_number<FLAGS.epoch_size:
    print('Loading training data')
    # Sample people and load new data
    image_paths, num_per_class = facenet.sample_people(dataset, FLAGS.people_per_batch, FLAGS.images_per_person)
    image_data = facenet.load_data(image_paths)
  
    nrof_examples_per_epoch = FLAGS.people_per_batch*FLAGS.images_per_person
    nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch/FLAGS.batch_size))
    
    print('Selecting suitable triplets for training')
    start_time = time.time()
    emb_list = []
    # Run a forward pass for the sampled images
    for i in xrange(nrof_batches_per_epoch):
      batch = facenet.get_batch(image_data, FLAGS.batch_size, i)
      feed_dict = { images_placeholder: batch, phase_train_placeholder: True }
      emb_list += sess.run([embeddings], feed_dict=feed_dict)
    emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
    # Select triplets based on the embeddings
    triplets, nrof_random_negs, nrof_triplets = facenet.select_training_triplets(emb_array, num_per_class, image_data)
    duration = time.time() - start_time
    print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (nrof_random_negs, nrof_triplets, duration))
    
    # Perform training on the selected triplets
    i = 0
    while i*FLAGS.batch_size<nrof_triplets*3 and batch_number<FLAGS.epoch_size:
      start_time = time.time()
      batch = facenet.get_triplet_batch(triplets, i)
      feed_dict = { images_placeholder: batch, phase_train_placeholder: True }
      if (batch_number%20==0):
        err, summary_str, _, step  = sess.run([loss, summary_op, train_op, global_step], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=step)
      else:
        err, _, step  = sess.run([loss, train_op, global_step], feed_dict=feed_dict)
      duration = time.time() - start_time
      print('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %2.3f' % (epoch, batch_number, FLAGS.epoch_size, duration, err))
      batch_number+=1
      i+=1
  return step
  
def validate_epoch(sess, dataset, epoch, images_placeholder, phase_train_placeholder, global_step, embeddings, loss, train_op, summary_op):
  people_per_batch = FLAGS.people_per_batch*4
  print('Loading validation data')
  # Sample people and load new data
  image_paths, num_per_class = facenet.sample_people(dataset, people_per_batch, FLAGS.images_per_person)
  image_data = facenet.load_data(image_paths)

  nrof_examples_per_epoch = people_per_batch*FLAGS.images_per_person
  nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch/FLAGS.batch_size))
  
  print('Selecting random triplets for validation')
  triplets, nrof_triplets = facenet.select_validation_triplets(num_per_class, people_per_batch, image_data)

  start_time = time.time()
  anchor_list = []
  positive_list = []
  negative_list = []
  loss_list = []
  # Run a forward pass for the sampled images
  print('Running forward pass on images')
  for i in xrange(nrof_batches_per_epoch):
    batch = facenet.get_triplet_batch(triplets, i)
    feed_dict = { images_placeholder: batch, phase_train_placeholder: True }
    emb_x, loss_x = sess.run([embeddings, loss], feed_dict=feed_dict)
    nrof_batch_triplets = emb_x.shape[0]/3
    anchor_list.append(emb_x[(0*nrof_batch_triplets):(1*nrof_batch_triplets),:])
    positive_list.append(emb_x[(1*nrof_batch_triplets):(2*nrof_batch_triplets),:])
    negative_list.append(emb_x[(2*nrof_batch_triplets):(3*nrof_batch_triplets),:])
    loss_list.append(loss_x)
  anchor = np.vstack(anchor_list)
  positive = np.vstack(positive_list)
  negative = np.vstack(negative_list)
  duration = time.time() - start_time
  
  thresholds = np.arange(0, 4, 0.01)
  embeddings1 = np.vstack([anchor, anchor])
  embeddings2 = np.vstack([positive, negative])
  actual_issame = [True]*anchor.shape[0] + [False]*anchor.shape[0]
  tpr, fpr, accuracy, predict_issame, dist = facenet.calculate_roc(thresholds, embeddings1, embeddings2, actual_issame)
  print('Epoch: [%d]\tTime %.3f\ttripErr %2.3f\taccuracy %1.3f' % (epoch, duration, np.mean(loss_list), np.max(accuracy)))

if __name__ == '__main__':
  tf.app.run()
