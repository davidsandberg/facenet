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


def filter_dataset(dataset, data_filename, percentile,
                   min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(
            distance_to_center, percentile)
        indices = np.where(distance_to_center >=
                           distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(
                    filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold


def save_variables_and_metagraph(
        sess,
        saver,
        summary_writer,
        model_dir: str,
        model_name: str,
        step: int):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables',
                      simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph',
                      simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def validate(
        args,
        sess,
        epoch,
        image_list,
        label_list,
        enqueue_op,
        image_paths_placeholder,
        labels_placeholder,
        control_placeholder,
        phase_train_placeholder,
        batch_size_placeholder,
        stat,
        loss,
        regularization_losses,
        cross_entropy_mean,
        accuracy,
        validate_every_n_epochs,
        use_fixed_image_standardization):

    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // lfw_batch_size
    nrof_images = nrof_batches * lfw_batch_size

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list[:nrof_images]), 1)
    image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]), 1)
    control_array = np.ones_like(
        labels_array,
        np.int32) * facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    sess.run(enqueue_op,
             {image_paths_placeholder: image_paths_array,
              labels_placeholder: labels_array,
              control_placeholder: control_array})

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False,
                     batch_size_placeholder: lfw_batch_size}
        loss_, cross_entropy_mean_, accuracy_ = sess.run(
            [loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (
            loss_, cross_entropy_mean_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch - 1) // validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_xent_loss'][val_index] = np.mean(xent_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' % (
        epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))


def evaluate(
        sess,
        enqueue_op,
        image_paths_placeholder,
        labels_placeholder,
        phase_train_placeholder,
        batch_size_placeholder,
        control_placeholder,
        embeddings,
        labels,
        image_paths,
        actual_issame,
        batch_size,
        nrof_folds,
        log_dir,
        step,
        summary_writer,
        stat,
        epoch,
        distance_metric,
        subtract_mean,
        use_flipped_images,
        use_fixed_image_standardization):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    # nrof_pairs * nrof_images_per_pair
    nrof_embeddings = len(actual_issame) * 2
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
    image_paths_array = np.expand_dims(
        np.repeat(np.array(image_paths), nrof_flips), 1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array) * \
            facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2) * facenet.FLIP
    sess.run(enqueue_op,
             {image_paths_placeholder: image_paths_array,
              labels_placeholder: labels_array,
              control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False,
                     batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the
        # images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(
        nrof_images)), 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(
        embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['lfw_accuracy'][epoch - 1] = np.mean(accuracy)
    stat['lfw_valrate'][epoch - 1] = val
