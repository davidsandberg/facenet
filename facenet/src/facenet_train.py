"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from subprocess import Popen, PIPE
import os.path
import time

import tensorflow as tf
import numpy as np
import importlib
import facenet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('logs_base_dir', '~/logs/facenet',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('models_base_dir', '~/models/facenet',
                           """Directory where to write trained models and checkpoints.""")
tf.app.flags.DEFINE_string('model_name', '',
                           """Model directory name. Used when continuing training of an existing model. Leave empty to train new model.""")
tf.app.flags.DEFINE_string('data_dir', '~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned',
                           """Path to the data directory containing aligned face patches. Multiple directories are separated with colon.""")
tf.app.flags.DEFINE_string('model_def', 'models.nn4',
                           """Model definition. Points to a module containing the definition of the inference graph.""")
tf.app.flags.DEFINE_integer('max_nrof_epochs', 500,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('checkpoint_period', 10,
                            """The number of epochs between checkpoints""")
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
tf.app.flags.DEFINE_integer('epoch_size', 1000,
                            """Number of batches per epoch.""")
tf.app.flags.DEFINE_float('alpha', 0.2,
                          """Positive to negative triplet distance margin.""")
tf.app.flags.DEFINE_boolean('random_crop', False,
                          """Performs random cropping of training images. If false, the center image_size pixels from the training images are used.
                          If the size of the images in the data directory is equal to image_size no cropping is performed""")
tf.app.flags.DEFINE_boolean('random_flip', False,
                          """Performs random horizontal flipping of training images.""")
tf.app.flags.DEFINE_string('pool_type', 'MAX',
                          """The type of pooling to use for some of the inception layers {'MAX', 'L2'}.""")
tf.app.flags.DEFINE_boolean('use_lrn', False,
                          """Enables Local Response Normalization after the first layers of the inception network.""")
tf.app.flags.DEFINE_float('keep_probability', 1.0,
                          """Keep probability of dropout for the fully connected layer(s).""")
tf.app.flags.DEFINE_string('optimizer', 'ADAGRAD',
                          """The optimization algorithm to use {'ADAGRAD', 'ADADELTA', 'ADAM'}.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """Exponential decay for tracking of training parameters.""")
tf.app.flags.DEFINE_float('train_set_fraction', 0.9,
                          """Fraction of the data set that is used for training.""")
tf.app.flags.DEFINE_string('split_mode', 'SPLIT_CLASSES',
                           """Defines the method used to split the data set into a train and test set { SPLIT_CLASSES, SPLIT_IMAGES }""")
tf.app.flags.DEFINE_integer('seed', 666, """Random seed.""")

network = importlib.import_module(FLAGS.model_def, 'inference')

def main(argv=None):  # pylint: disable=unused-argument
  
    if FLAGS.model_name:
        subdir = FLAGS.model_name
        preload_model = True
    else:
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        preload_model = False
    log_dir = os.path.join(os.path.expanduser(FLAGS.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.mkdir(log_dir)
    model_dir = os.path.join(os.path.expanduser(FLAGS.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.mkdir(model_dir)

    # Store some git revision info in a text file in the log directory        
    src_path,_ = os.path.split(os.path.realpath(__file__))
    store_training_info(src_path, log_dir, ' '.join(argv))
    
    np.random.seed(seed=FLAGS.seed)
    dataset = facenet.get_dataset(FLAGS.data_dir)
    train_set, validation_set = facenet.split_dataset(dataset, FLAGS.train_set_fraction, FLAGS.split_mode)
    
    shuffle = np.arange((FLAGS.images_per_person-1)*FLAGS.people_per_batch*4)
    np.random.shuffle(shuffle)
    
    print('Model directory: %s' % model_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')

        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Build the inference graph
        embeddings = network.inference(images_placeholder, FLAGS.pool_type, FLAGS.use_lrn, 
                                       FLAGS.keep_probability, phase_train=phase_train_placeholder)

        # Split example embeddings into anchor, positive and negative
        anchor, positive, negative = tf.split(0, 3, embeddings)

        # Calculate triplet loss
        loss = facenet.triplet_loss(anchor, positive, negative, FLAGS.alpha)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op, _ = facenet.train(loss, global_step, FLAGS.optimizer, FLAGS.learning_rate, FLAGS.moving_average_decay)

        # Create a saver
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

        with sess.as_default():

            if preload_model:
                ckpt = tf.train.get_checkpoint_state(model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError('Checkpoint not found')

            # Training and validation loop
            for epoch in range(FLAGS.max_nrof_epochs):
                # Train for one epoch
                step = train(sess, train_set, epoch, images_placeholder, phase_train_placeholder,
                             global_step, embeddings, loss, train_op, summary_op, summary_writer)
                # Test on validation set
                validate(sess, validation_set, epoch, images_placeholder, phase_train_placeholder,
                         global_step, embeddings, loss, 'validation', summary_writer, shuffle)
                # Test on training set
                validate(sess, train_set, epoch, images_placeholder, phase_train_placeholder,
                         global_step, embeddings, loss, 'training', summary_writer, shuffle)

                if (epoch % FLAGS.checkpoint_period == 0) or (epoch==FLAGS.max_nrof_epochs-1):
                  # Save the model checkpoint
                  print('Saving checkpoint')
                  checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                  saver.save(sess, checkpoint_path, global_step=step)
                  
                # Save the model if it hasn't been saved before
                graphdef_dir = os.path.join(model_dir, 'graphdef')
                graphdef_filename = 'graph_def.pb'
                if (not os.path.exists(os.path.join(graphdef_dir, graphdef_filename))):
                    print('Saving graph definition')
                    tf.train.write_graph(sess.graph_def, graphdef_dir, graphdef_filename, False)


def train(sess, dataset, epoch, images_placeholder, phase_train_placeholder,
          global_step, embeddings, loss, train_op, summary_op, summary_writer):
    batch_number = 0
    while batch_number < FLAGS.epoch_size:
        print('Loading training data')
        # Sample people and load new data
        image_paths, num_per_class = facenet.sample_people(dataset, FLAGS.people_per_batch, FLAGS.images_per_person)
        image_data = facenet.load_data(image_paths, FLAGS.random_crop, FLAGS.random_flip, FLAGS.image_size)

        print('Selecting suitable triplets for training')
        start_time = time.time()
        emb_list = []
        # Run a forward pass for the sampled images
        nrof_examples_per_epoch = FLAGS.people_per_batch * FLAGS.images_per_person
        nrof_batches_per_epoch = int(np.floor(nrof_examples_per_epoch / FLAGS.batch_size))
        for i in xrange(nrof_batches_per_epoch):
            batch = facenet.get_batch(image_data, FLAGS.batch_size, i)
            feed_dict = {images_placeholder: batch, phase_train_placeholder: True}
            emb_list += sess.run([embeddings], feed_dict=feed_dict)
        emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
        # Select triplets based on the embeddings
        triplets, nrof_random_negs, nrof_triplets = facenet.select_training_triplets(emb_array, num_per_class, 
                                                                                     image_data, FLAGS.people_per_batch, 
                                                                                     FLAGS.alpha)
        duration = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % (
        nrof_random_negs, nrof_triplets, duration))

        # Perform training on the selected triplets
        i = 0
        while i * FLAGS.batch_size < nrof_triplets * 3 and batch_number < FLAGS.epoch_size:
            start_time = time.time()
            batch = facenet.get_triplet_batch(triplets, i, FLAGS.batch_size)
            feed_dict = {images_placeholder: batch, phase_train_placeholder: True}
            err, _, step = sess.run([loss, train_op, global_step], feed_dict=feed_dict)
            if (batch_number % 20 == 0):
                summary_str, step = sess.run([summary_op, global_step], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %2.3f' %
                  (epoch, batch_number, FLAGS.epoch_size, duration, err))
            batch_number += 1
            i += 1
    return step


def validate(sess, dataset, epoch, images_placeholder, phase_train_placeholder,
             global_step, embeddings, loss, prefix_str, summary_writer, shuffle):
    nrof_people = FLAGS.people_per_batch * 4
    print('Loading %s data' % prefix_str)
    # Sample people and load new data
    image_paths, num_per_class = facenet.sample_people(dataset, nrof_people, FLAGS.images_per_person)
    image_data = facenet.load_data(image_paths, False, False, FLAGS.image_size)

    print('Selecting random triplets from %s set' % prefix_str)
    triplets, nrof_triplets = facenet.select_validation_triplets(num_per_class, nrof_people, image_data, FLAGS.batch_size, shuffle)

    start_time = time.time()
    anchor_list = []
    positive_list = []
    negative_list = []
    triplet_loss_list = []
    # Run a forward pass for the sampled images
    print('Running forward pass on %s set' % prefix_str)
    nrof_batches_per_epoch = nrof_triplets * 3 // FLAGS.batch_size
    for i in xrange(nrof_batches_per_epoch):
        batch = facenet.get_triplet_batch(triplets, i, FLAGS.batch_size)
        feed_dict = {images_placeholder: batch, phase_train_placeholder: False}
        emb, triplet_loss, step = sess.run([embeddings, loss, global_step], feed_dict=feed_dict)
        nrof_batch_triplets = emb.shape[0] / 3
        anchor_list.append(emb[(0 * nrof_batch_triplets):(1 * nrof_batch_triplets), :])
        positive_list.append(emb[(1 * nrof_batch_triplets):(2 * nrof_batch_triplets), :])
        negative_list.append(emb[(2 * nrof_batch_triplets):(3 * nrof_batch_triplets), :])
        triplet_loss_list.append(triplet_loss)
    anchor = np.vstack(anchor_list)
    positive = np.vstack(positive_list)
    negative = np.vstack(negative_list)
    duration = time.time() - start_time

    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = np.vstack([anchor, anchor])
    embeddings2 = np.vstack([positive, negative])
    actual_issame = np.asarray([True] * anchor.shape[0] + [False] * anchor.shape[0])
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, FLAGS.seed)
    print('Epoch: [%d]\tTime %.3f\ttripErr %2.3f\t%sAccuracy %1.3f+-%1.3f' % (
    epoch, duration, np.mean(triplet_loss_list), prefix_str, np.mean(accuracy), np.std(accuracy)))

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='{}_loss'.format(prefix_str), simple_value=np.mean(triplet_loss_list).astype(float))
    summary.value.add(tag='{}_accuracy'.format(prefix_str), simple_value=np.mean(accuracy))
    summary_writer.add_summary(summary, step)

    if False:
      facenet.plot_roc(fpr, tpr, 'NN4')

def store_training_info(src_path, log_dir, arg_string):
    # Get git hash 
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()

    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(log_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
      text_file.write('arguments: %s\n--------------------\n' % arg_string)
      text_file.write('git hash: %s\n--------------------\n' % git_hash)
      text_file.write('%s' % git_diff)
      
if __name__ == '__main__':
    tf.app.run()
