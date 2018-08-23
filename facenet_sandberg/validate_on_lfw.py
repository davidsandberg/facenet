"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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
import math
import os
import sys
import warnings
from enum import Enum, auto
from typing import List, Tuple, Union, cast

import numpy as np
import progressbar as pb
import tensorflow as tf
from facenet_sandberg import facenet, lfw
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import KFold
from tensorflow.python.ops import data_flow_ops

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class DistanceMetric(Enum):
    ANGULAR_DISTANCE = auto()
    EUCLIDEAN_SQUARED = auto()


def main(lfw_dir, model, lfw_pairs, use_flipped_images, subtract_mean,
         use_fixed_image_standardization, image_size=160, lfw_nrof_folds=10,
         distance_metric=0, lfw_batch_size=128):
    """Runs testing on dataset

    Arguments:
        lfw_dir {str} -- Path to the data directory containing aligned LFW face patches.
        model {str} -- Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file.
        lfw_pairs {str} -- The file containing the pairs to use for validation.
        use_flipped_images {bool} -- Concatenates embeddings for the image and its horizontally flipped counterpart.
        subtract_mean {bool} -- Subtract feature mean before calculating distance.
        use_fixed_image_standardization {bool} -- Performs fixed standardization of images.

    Keyword Arguments:
        image_size {int} -- [description] (default: {160})
        lfw_nrof_folds {int} -- Number of folds to use for cross validation. Mainly used for testing. (default: {10})
        distance_metric {int} -- Distance metric  0:euclidian, 1:cosine similarity. (default: {0})
        lfw_batch_size {int} -- Number of images to process in a batch in the LFW test set. (default: {128})
    """

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

            # Get the paths for the corresponding images
            paths, labels = lfw.get_paths(os.path.expanduser(lfw_dir), pairs)

            image_paths_placeholder = tf.placeholder(
                tf.string, shape=(None, 1), name='image_paths')
            labels_placeholder = tf.placeholder(
                tf.int32, shape=(None, 1), name='labels')
            batch_size_placeholder = tf.placeholder(
                tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(
                tf.int32, shape=(None, 1), name='control')
            phase_train_placeholder = tf.placeholder(
                tf.bool, name='phase_train')

            nrof_preprocess_threads = 4
            image_size = (image_size, image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(
                capacity=2000000, dtypes=[
                    tf.string, tf.int32, tf.int32], shapes=[
                    (1,), (1,), (1,)], shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder,
                                                             control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(
                eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

            # Load the model
            input_map = {
                'image_batch': image_batch,
                'label_batch': label_batch,
                'phase_train': phase_train_placeholder}
            facenet.load_model(model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            evaluate(
                sess,
                eval_enqueue_op,
                image_paths_placeholder,
                labels_placeholder,
                phase_train_placeholder,
                batch_size_placeholder,
                control_placeholder,
                embeddings,
                label_batch,
                paths,
                labels,
                lfw_batch_size,
                lfw_nrof_folds,
                distance_metric,
                subtract_mean,
                use_flipped_images,
                use_fixed_image_standardization)


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
        distance_metric,
        subtract_mean,
        use_flipped_images,
        use_fixed_image_standardization):
    # Run forward pass to calculate embeddings
    widgets = ['Scoring', pb.Percentage(), ' ',
               pb.Bar(marker=pb.Bar()), ' ', pb.ETA()]

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

    timer = pb.ProgressBar(
        widgets=widgets, maxval=int(
            nrof_batches + 1)).start()
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False,
                     batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        timer.update(i + 1)
    timer.finish()
    embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the
        # images
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    accuracy, recall, precision = score(embeddings,
                                        np.asarray(actual_issame),
                                        nrof_folds,
                                        'ANGULAR_DISTANCE',
                                        'ACCURACY',
                                        subtract_mean,
                                        False,
                                        0,
                                        4,
                                        0.01)
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')

    # assert np.array_equal(lab_array, np.arange(
    #     nrof_images)), 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    # tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(
    # embeddings, actual_issame, nrof_folds=nrof_folds,
    # distance_metric=distance_metric, subtract_mean=subtract_mean)

    # print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    # auc = metrics.auc(fpr, tpr)
    # print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)


def score(embeddings: np.ndarray,
          labels: np.ndarray,
          num_folds: int,
          distance_metric: DistanceMetric,
          threshold_metric: str,
          subtract_mean: bool,
          divide_stddev: bool,
          threshold_start: float,
          threshold_end: float,
          threshold_step: float) -> Tuple[np.float, np.float, np.float]:
    thresholds = np.arange(threshold_start, threshold_end, threshold_step)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    accuracy, recall, precision = _score_k_fold(thresholds,
                                                embeddings1,
                                                embeddings2,
                                                labels,
                                                num_folds,
                                                threshold_metric,
                                                subtract_mean,
                                                divide_stddev)
    return np.mean(accuracy), np.mean(recall), np.mean(precision)


def _score_k_fold(thresholds: np.ndarray,
                  embeddings1: np.ndarray,
                  embeddings2: np.ndarray,
                  labels: np.ndarray,
                  num_folds: int,
                  threshold_metric: str,
                  subtract_mean: bool,
                  divide_stddev: bool) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:
    k_fold = KFold(n_splits=num_folds, shuffle=True)
    accuracy = np.zeros((num_folds))
    recall = np.zeros((num_folds))
    precision = np.zeros((num_folds))
    splits = k_fold.split(np.arange(len(labels)))
    for fold_idx, (train_set, test_set) in enumerate(splits):
        train_embeddings = np.concatenate([embeddings1[train_set],
                                           embeddings2[train_set]])
        mean = np.mean(train_embeddings, axis=0) if subtract_mean else 0.0
        stddev = np.std(train_embeddings, axis=0) if divide_stddev else 1.0
        dist = _distance_between_embeddings((embeddings1 - mean) / stddev,
                                            (embeddings2 - mean) / stddev)
        best_threshold = _calculate_best_threshold(thresholds,
                                                   dist[train_set],
                                                   labels[train_set],
                                                   threshold_metric)
        predictions = np.less(dist[test_set], best_threshold)
        accuracy[fold_idx] = accuracy_score(labels[test_set], predictions)
        recall[fold_idx] = recall_score(labels[test_set], predictions)
        precision[fold_idx] = precision_score(labels[test_set], predictions)
    return accuracy, recall, precision


def _distance_between_embeddings(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray) -> np.ndarray:
    # if distance_metric == DistanceMetric.EUCLIDEAN_SQUARED:
    #     return np.square(
    #         paired_distances(
    #             embeddings1,
    #             embeddings2,
    #             metric='euclidean'))
    # elif distance_metric == DistanceMetric.ANGULAR_DISTANCE:
        # Angular Distance: https://en.wikipedia.org/wiki/Cosine_similarity
    similarity = 1 - paired_distances(
        embeddings1,
        embeddings2,
        metric='cosine')
    return np.arccos(similarity) / math.pi


def _calculate_best_threshold(thresholds: np.ndarray,
                              dist: np.ndarray,
                              labels: np.ndarray,
                              threshold_metric: str) -> np.float:

    if threshold_metric == 'ACCURACY':
        threshold_score = accuracy_score
    elif threshold_metric == 'PRECISION':
        threshold_score = precision_score
    elif threshold_metric == 'RECALL':
        threshold_score = recall_score
    threshold_scores = np.zeros((len(thresholds)))
    for threshold_idx, threshold in enumerate(thresholds):
        predictions = np.less(dist, threshold)
        threshold_scores[threshold_idx] = threshold_score(labels, predictions)
    best_threshold_index = np.argmax(threshold_scores)
    return thresholds[best_threshold_index]


def parse_arguments(argv):
    """Argument parser

    Arguments:
        argv {} -- arguments

    Returns:
        {} -- parsed arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'lfw_dir',
        type=str,
        help='Path to the data directory containing aligned LFW face patches.',
        default='/Users/armanrahman/datasets/eame_test_facenet_old')
    parser.add_argument(
        '--lfw_batch_size',
        type=int,
        help='Number of images to process in a batch in the LFW test set.',
        default=100)
    parser.add_argument(
        'model',
        type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
        default='/Users/armanrahman/models/facenet_model.pb')
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=160)
    parser.add_argument(
        '--lfw_pairs',
        type=str,
        help='The file containing the pairs to use for validation.',
        default='/Users/armanrahman/datasets/eame_test_pairs_facenet.txt')
    parser.add_argument(
        '--lfw_nrof_folds',
        type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.',
        default=10)
    parser.add_argument(
        '--distance_metric',
        type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.',
        default=0)
    parser.add_argument(
        '--use_flipped_images',
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
        action='store_true')
    parser.add_argument(
        '--subtract_mean',
        help='Subtract feature mean before calculating distance.',
        action='store_true')
    parser.add_argument(
        '--use_fixed_image_standardization',
        help='Performs fixed standardization of images.',
        action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args:
        main(
            args.lfw_dir,
            args.model,
            args.lfw_pairs,
            args.use_flipped_images,
            args.subtract_mean,
            args.use_fixed_image_standardization,
            args.image_size,
            args.lfw_nrof_folds,
            args.distance_metric,
            args.lfw_batch_size)
