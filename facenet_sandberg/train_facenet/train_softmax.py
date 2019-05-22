"""Training a face recognizer with TensorFlow using softmax cross entropy loss
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

from __future__ import absolute_import, division, print_function

import argparse
import importlib
import math
import os.path
import random
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from facenet_sandberg import facenet, lfw
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, data_flow_ops

# parser.add_argument(
#     '--optimizer',
#     type=str,
#     choices=[
#         'ADAGRAD',
#         'ADADELTA',
#         'ADAM',
#         'RMSPROP',
#         'MOM'],
#     help='The optimization algorithm to use',
#     default='ADAGRAD')


def main(
        pretrained_model: str,
        logs_base_dir: str='~/logs/facenet',
        models_base_dir: str ='~/models/facenet',
        gpu_memory_fraction: float=1.0,
        data_dir: str='~/datasets/casia/casia_maxpy_mtcnnalign_182_160',
        model_def: str='models.inception_resnet_v1',
        max_nrof_epochs: int=500,
        batch_size: int=100,
        image_size: int=160,
        epoch_size: int=1000,
        embedding_size: int=128,
        random_crop: bool=False,
        random_flip: bool=False,
        random_rotate: bool=False,
        use_fixed_image_standardization: bool=False,
        keep_probability: float=1.0,
        weight_decay: float=0.0,
        center_loss_factor: float=0.0,
        center_loss_alfa: float=0.95,
        prelogits_norm_loss_factor: float=0.0,
        prelogits_norm_p: float=1.0,
        prelogits_hist_max: float=10.0,
        optimizer: str='ADAGRAD',
        learning_rate: float=0.1,
        learning_rate_decay_epochs: int=100,
        learning_rate_decay_factor: float=1.0,
        moving_average_decay: float=0.9999,
        seed: int=666,
        nrof_preprocess_threads: int=4,
        log_histograms: bool=False,
        learning_rate_schedule_file: str='data/learning_rate_schedule.txt',
        filter_filename: str='',
        filter_percentile: float=100.0,
        filter_min_nrof_images_per_class: int=0,
        validate_every_n_epochs: int=5,
        validation_set_split_ratio: float=0.0,
        min_nrof_val_images_per_class: int=0,
        lfw_pairs: str='data/pairs.txt',
        lfw_dir: str='',
        lfw_batch_size: int=100,
        lfw_nrof_folds: int=10,
        lfw_distance_metric: int=0,
        lfw_use_flipped_images: bool=False,
        lfw_subtract_mean: bool=False):
    """Train with softmax

    Arguments:
        pretrained_model {str} -- Load a pretrained model before training starts.

    Keyword Arguments:
        logs_base_dir {str} -- Directory where to write event logs. (default: {'~/logs/facenet'})
        models_base_dir {str} -- Directory where to write trained models and checkpoints. (default: {'~/models/facenet'})
        gpu_memory_fraction {float} -- Upper bound on the amount of GPU memory that will be used by the process. (default: {1.0})
        data_dir {str} -- Path to the data directory containing aligned face patches. (default: {'~/datasets/casia/casia_maxpy_mtcnnalign_182_160'})
        model_def {str} -- Model definition. Points to a module containing the definition of the inference graph. (default: {'models.inception_resnet_v1'})
        max_nrof_epochs {int} -- Number of epochs to run. (default: {500})
        batch_size {int} -- Number of images to process in a batch. (default: {100})
        image_size {int} -- Image size (height, width) in pixels. (default: {160})
        epoch_size {int} -- Number of batches per epoch. (default: {1000})
        embedding_size {int} -- Dimensionality of the embedding. (default: {128})
        random_crop {bool} -- Performs random cropping of training images. If false, the center image_size pixels from the training images are used. If the size of the images in the data directory is equal to image_size no cropping is performed (default: {False})
        random_flip {bool} -- Performs random horizontal flipping of training images. (default: {False})
        random_rotate {bool} -- Performs random rotations of training images. (default: {False})
        use_fixed_image_standardization {bool} -- Performs fixed standardization of images. (default: {False})
        keep_probability {float} -- Keep probability of dropout for the fully connected layer(s). (default: {1.0})
        weight_decay {float} -- L2 weight regularization. (default: {0.0})
        center_loss_factor {float} -- Center loss factor. (default: {0.0})
        center_loss_alfa {float} -- Center update rate for center loss. (default: {0.95})
        prelogits_norm_loss_factor {float} -- Loss based on the norm of the activations in the prelogits layer. (default: {0.0})
        prelogits_norm_p {float} -- Norm to use for prelogits norm loss. (default: {1.0})
        prelogits_hist_max {float} -- The max value for the prelogits histogram. (default: {10.0})
        optimizer {str} -- The optimization algorithm to use (default: {'ADAGRAD'})
        learning_rate {float} -- Initial learning rate. If set to a negative value a learning rate schedule can be specified in the file "learning_rate_schedule.txt" (default: {0.1})
        learning_rate_decay_epochs {int} -- Number of epochs between learning rate decay. (default: {100})
        learning_rate_decay_factor {float} -- Learning rate decay factor. (default: {1.0})
        moving_average_decay {float} -- Exponential decay for tracking of training parameters. (default: {0.9999})
        seed {int} -- Random seed. (default: {666})
        nrof_preprocess_threads {int} -- Number of preprocessing (data loading and augmentation) threads. (default: {4})
        log_histograms {bool} -- Enables logging of weight/bias histograms in tensorboard. (default: {False})
        learning_rate_schedule_file {str} -- File containing the learning rate schedule that is used when learning_rate is set to to -1. (default: {'data/learning_rate_schedule.txt'})
        filter_filename {str} -- File containing image data used for dataset filtering (default: {''})
        filter_percentile {float} -- Keep only the percentile images closed to its class center (default: {100.0})
        filter_min_nrof_images_per_class {int} -- Keep only the classes with this number of examples or more (default: {0})
        validate_every_n_epochs {int} -- Number of epoch between validation (default: {5})
        validation_set_split_ratio {float} -- The ratio of the total dataset to use for validation (default: {0.0})
        min_nrof_val_images_per_class {int} -- Classes with fewer images will be removed from the validation set (default: {0})
        lfw_pairs {str} -- The file containing the pairs to use for validation. (default: {'data/pairs.txt'})
        lfw_dir {str} -- Path to the data directory containing aligned face patches. (default: {''})
        lfw_batch_size {int} -- Number of images to process in a batch in the LFW test set. (default: {100})
        lfw_nrof_folds {int} -- Number of folds to use for cross validation. Mainly used for testing. (default: {10})
        lfw_distance_metric {int} -- Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance. (default: {0})
        lfw_use_flipped_images {bool} -- Concatenates embeddings for the image and its horizontally flipped counterpart. (default: {False})
        lfw_subtract_mean {bool} -- Subtract feature mean before calculating distance. (default: {False})

    Returns:
        [type] -- [description]
    """

    network = importlib.import_module(model_def)
    image_size = (image_size, image_size)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(os.path.expanduser(models_base_dir), subdir)
    os.makedirs(model_dir, exist_ok=True)

    stat_file_name = os.path.join(log_dir, 'stat.h5')

    # Write arguments to a text file
    # facenet.write_arguments_to_file(
    #     args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=seed)
    random.seed(seed)
    dataset = facenet.get_dataset(data_dir)
    if filter_filename:
        dataset = filter_dataset(
            dataset,
            os.path.expanduser(
                filter_filename),
            filter_percentile,
            filter_min_nrof_images_per_class)

    if validation_set_split_ratio > 0.0:
        train_set, val_set = facenet.split_dataset(
            dataset, validation_set_split_ratio, min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    val_image_list, val_label_list = facenet.get_image_paths_and_labels(
        val_set)
    assert len(image_list) > 0, 'The training set should not be empty'

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if pretrained_model:
        pretrained_model = os.path.expanduser(pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if lfw_dir:
        print('LFW directory: %s' % lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, lfw_labels = lfw.get_paths(
            os.path.expanduser(lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        global_step = tf.Variable(0, trainable=False)

        # Create a queue that produces indices into the image_list and
        # label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(
            range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(
            batch_size * epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(
            tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(
            tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(
            tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(
            tf.int32, shape=(None, 1), name='control')

        nrof_preprocess_threads = 4
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.string,
                                                      tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many(
            [image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        image_batch, label_batch = facenet.create_input_pipeline(
            input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))

        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                         weight_decay=weight_decay)
        logits = slim.fully_connected(
            prelogits,
            len(train_set),
            activation_fn=None,
            weights_initializer=slim.initializers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(
                weight_decay),
            scope='Logits',
            reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(
            tf.norm(
                tf.abs(prelogits) + eps,
                ord=prelogits_norm_p,
                axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             prelogits_norm * prelogits_norm_loss_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(
            prelogits, label_batch, center_loss_alfa, nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             prelogits_center_loss * center_loss_factor)

        learning_rate = tf.train.exponential_decay(
            learning_rate_placeholder,
            global_step,
            learning_rate_decay_epochs *
            epoch_size,
            learning_rate_decay_factor,
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(
            tf.equal(
                tf.argmax(
                    logits, 1), tf.cast(
                    label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] +
                              regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters
        train_op = facenet.train(
            total_loss,
            global_step,
            optimizer,
            learning_rate,
            moving_average_decay,
            tf.global_variables(),
            log_histograms)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            nrof_steps = max_nrof_epochs * epoch_size
            # Validate every validate_every_n_epochs as well as in the last
            # epoch
            nrof_val_samples = int(
                math.ceil(max_nrof_epochs / validate_every_n_epochs))
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((max_nrof_epochs,), np.float32),
                'time_train': np.zeros((max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((max_nrof_epochs, 1000), np.float32),
            }
            for epoch in range(1, max_nrof_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = train(
                    args,
                    sess,
                    epoch,
                    image_list,
                    label_list,
                    index_dequeue_op,
                    enqueue_op,
                    image_paths_placeholder,
                    labels_placeholder,
                    learning_rate_placeholder,
                    phase_train_placeholder,
                    batch_size_placeholder,
                    control_placeholder,
                    global_step,
                    total_loss,
                    train_op,
                    summary_op,
                    summary_writer,
                    regularization_losses,
                    learning_rate_schedule_file,
                    stat,
                    cross_entropy_mean,
                    accuracy,
                    learning_rate,
                    prelogits,
                    prelogits_center_loss,
                    random_rotate,
                    random_crop,
                    random_flip,
                    prelogits_norm,
                    prelogits_hist_max,
                    use_fixed_image_standardization)
                stat['time_train'][epoch - 1] = time.time() - t

                if not cont:
                    break

                t = time.time()
                if len(val_image_list) > 0 and (
                    (epoch - 1) %
                    validate_every_n_epochs == validate_every_n_epochs -
                        1 or epoch == max_nrof_epochs):
                    validate(
                        args,
                        sess,
                        epoch,
                        val_image_list,
                        val_label_list,
                        enqueue_op,
                        image_paths_placeholder,
                        labels_placeholder,
                        control_placeholder,
                        phase_train_placeholder,
                        batch_size_placeholder,
                        stat,
                        total_loss,
                        regularization_losses,
                        cross_entropy_mean,
                        accuracy,
                        validate_every_n_epochs,
                        use_fixed_image_standardization)
                stat['time_validate'][epoch - 1] = time.time() - t

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(
                    sess, saver, summary_writer, model_dir, subdir, epoch)

                # Evaluate on LFW
                t = time.time()
                if lfw_dir:
                    evaluate(
                        sess,
                        enqueue_op,
                        image_paths_placeholder,
                        labels_placeholder,
                        phase_train_placeholder,
                        batch_size_placeholder,
                        control_placeholder,
                        embeddings,
                        label_batch,
                        lfw_paths,
                        lfw_labels,
                        lfw_batch_size,
                        lfw_nrof_folds,
                        log_dir,
                        step,
                        summary_writer,
                        stat,
                        epoch,
                        lfw_distance_metric,
                        lfw_subtract_mean,
                        lfw_use_flipped_images,
                        use_fixed_image_standardization)
                stat['time_evaluate'][epoch - 1] = time.time() - t

                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.iteritems():
                        f.create_dataset(key, data=value)

    return model_dir


def train(
        args,
        sess,
        epoch,
        image_list,
        label_list,
        index_dequeue_op,
        enqueue_op,
        image_paths_placeholder,
        labels_placeholder,
        learning_rate_placeholder,
        phase_train_placeholder,
        batch_size_placeholder,
        control_placeholder,
        step,
        loss,
        train_op,
        summary_op,
        summary_writer,
        reg_losses,
        learning_rate_schedule_file,
        stat,
        cross_entropy_mean,
        accuracy,
        learning_rate,
        prelogits,
        prelogits_center_loss,
        random_rotate,
        random_crop,
        random_flip,
        prelogits_norm,
        prelogits_hist_max,
        use_fixed_image_standardization):
    batch_number = 0

    if learning_rate > 0.0:
        lr = learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(
            learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + \
        facenet.RANDOM_FLIP * random_flip + \
        facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op,
             {image_paths_placeholder: image_paths_array,
              labels_placeholder: labels_array,
              control_placeholder: control_array})

    # Training loop
    train_time = 0
    while batch_number < epoch_size:
        start_time = time.time()
        feed_dict = {
            learning_rate_placeholder: lr,
            phase_train_placeholder: True,
            batch_size_placeholder: batch_size}
        tensor_list = [
            loss,
            train_op,
            step,
            reg_losses,
            prelogits,
            cross_entropy_mean,
            learning_rate,
            prelogits_norm,
            accuracy,
            prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(
                tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1,
                               :] += np.histogram(np.minimum(np.abs(prelogits_),
                                                             prelogits_hist_max),
                                                  bins=1000,
                                                  range=(0.0,
                                                         prelogits_hist_max))[0]

        duration = time.time() - start_time
        print(
            'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
            (epoch,
             batch_number +
             1,
             epoch_size,
             duration,
             loss_,
             cross_entropy_mean_,
             np.sum(reg_losses_),
                accuracy_,
                lr_,
                center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--logs_base_dir',
        type=str,
        help='Directory where to write event logs.',
        default='~/logs/facenet')
    parser.add_argument(
        '--models_base_dir',
        type=str,
        help='Directory where to write trained models and checkpoints.',
        default='~/models/facenet')
    parser.add_argument(
        '--gpu_memory_fraction',
        type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument(
        '--model_def',
        type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.',
        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Number of images to process in a batch.',
        default=90)
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument(
        '--random_crop',
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
        'If the size of the images in the data directory is equal to image_size no cropping is performed',
        action='store_true')
    parser.add_argument(
        '--random_flip',
        help='Performs random horizontal flipping of training images.',
        action='store_true')
    parser.add_argument(
        '--random_rotate',
        help='Performs random rotations of training images.',
        action='store_true')
    parser.add_argument(
        '--use_fixed_image_standardization',
        help='Performs fixed standardization of images.',
        action='store_true')
    parser.add_argument(
        '--keep_probability',
        type=float,
        help='Keep probability of dropout for the fully connected layer(s).',
        default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument(
        '--center_loss_alfa',
        type=float,
        help='Center update rate for center loss.',
        default=0.95)
    parser.add_argument(
        '--prelogits_norm_loss_factor',
        type=float,
        help='Loss based on the norm of the activations in the prelogits layer.',
        default=0.0)
    parser.add_argument(
        '--prelogits_norm_p',
        type=float,
        help='Norm to use for prelogits norm loss.',
        default=1.0)
    parser.add_argument(
        '--prelogits_hist_max',
        type=float,
        help='The max value for the prelogits histogram.',
        default=10.0)
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=[
            'ADAGRAD',
            'ADADELTA',
            'ADAM',
            'RMSPROP',
            'MOM'],
        help='The optimization algorithm to use',
        default='ADAGRAD')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"',
        default=0.1)
    parser.add_argument(
        '--learning_rate_decay_epochs',
        type=int,
        help='Number of epochs between learning rate decay.',
        default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument(
        '--moving_average_decay',
        type=float,
        help='Exponential decay for tracking of training parameters.',
        default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument(
        '--nrof_preprocess_threads',
        type=int,
        help='Number of preprocessing (data loading and augmentation) threads.',
        default=4)
    parser.add_argument(
        '--log_histograms',
        help='Enables logging of weight/bias histograms in tensorboard.',
        action='store_true')
    parser.add_argument(
        '--learning_rate_schedule_file',
        type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
        default='data/learning_rate_schedule.txt')
    parser.add_argument(
        '--filter_filename',
        type=str,
        help='File containing image data used for dataset filtering',
        default='')
    parser.add_argument(
        '--filter_percentile',
        type=float,
        help='Keep only the percentile images closed to its class center',
        default=100.0)
    parser.add_argument(
        '--filter_min_nrof_images_per_class',
        type=int,
        help='Keep only the classes with this number of examples or more',
        default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
                        help='Number of epoch between validation', default=5)
    parser.add_argument(
        '--validation_set_split_ratio',
        type=float,
        help='The ratio of the total dataset to use for validation',
        default=0.0)
    parser.add_argument(
        '--min_nrof_val_images_per_class',
        type=float,
        help='Classes with fewer images will be removed from the validation set',
        default=0)

    # Parameters for validation on LFW
    parser.add_argument(
        '--lfw_pairs',
        type=str,
        help='The file containing the pairs to use for validation.',
        default='data/pairs.txt')
    parser.add_argument(
        '--lfw_dir',
        type=str,
        help='Path to the data directory containing aligned face patches.',
        default='')
    parser.add_argument(
        '--lfw_batch_size',
        type=int,
        help='Number of images to process in a batch in the LFW test set.',
        default=100)
    parser.add_argument(
        '--lfw_nrof_folds',
        type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.',
        default=10)
    parser.add_argument(
        '--lfw_distance_metric',
        type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.',
        default=0)
    parser.add_argument(
        '--lfw_use_flipped_images',
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
        action='store_true')
    parser.add_argument(
        '--lfw_subtract_mean',
        help='Subtract feature mean before calculating distance.',
        action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
