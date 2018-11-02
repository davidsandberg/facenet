import json
import math
from argparse import ArgumentParser, Namespace
from enum import Enum
from os.path import exists, join
from typing import List, Tuple, Union, cast

import docker
import numpy as np
from facenet_sandberg import utils
from facenet_sandberg.common_types import *
from facenet_sandberg.inference import identifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import KFold

FaceVector = List[float]
Match = Tuple[str, int, int]
Mismatch = Tuple[str, int, str, int]
Pair = Union[Match, Mismatch]
Path = Tuple[str, str]
Label = bool


def evaluate(embeddings: np.ndarray,
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
                                                distance_metric,
                                                threshold_metric,
                                                subtract_mean,
                                                divide_stddev)
    return np.mean(accuracy), np.mean(recall), np.mean(precision)


def _read_pairs(pairs_filename: str) -> Tuple[List[Pair], int, int]:
    pairs = []
    with open(pairs_filename, 'r') as pair_file:
        num_sets, num_matches_mismatches = [int(i)
                                            for i in next(pair_file).split()]
        for line_num, line in enumerate(pair_file):
            pair = cast(Pair, tuple([int(i) if i.isdigit() else i
                                     for i in line.strip().split()]))
            pairs.append(pair)
    return pairs, num_sets, num_matches_mismatches


def _get_paths_and_labels(image_dir: str,
                          pairs: List[Pair]) -> Tuple[List[Path], List[Label]]:
    paths = []
    labels = []
    for pair in pairs:
        _add_extension = (
            lambda rel_image_path,
            image_dir: '{}.jpg'.format(rel_image_path) if exists(
                join(
                    image_dir,
                    '{}.jpg'.format(rel_image_path))) else '{}.png'.format(rel_image_path))
        if len(pair) == 3:
            person, image_num_0, image_num_1 = cast(Match, pair)
            rel_image_path_no_ext = join(
                person, '{}_{}'.format(
                    person, '%04d' %
                    int(image_num_0)))
            rel_image_path_0 = _add_extension(rel_image_path_no_ext, image_dir)
            rel_image_path_no_ext = join(
                person, '{}_{}'.format(
                    person, '%04d' %
                    int(image_num_1)))
            rel_image_path_1 = _add_extension(rel_image_path_no_ext, image_dir)
            is_same_person = True
        elif len(pair) == 4:
            person_0, image_num_0, person_1, image_num_1 = cast(Mismatch, pair)
            rel_image_path_no_ext = join(
                person_0, '{}_{}'.format(
                    person_0, '%04d' %
                    int(image_num_0)))
            rel_image_path_0 = _add_extension(rel_image_path_no_ext, image_dir)
            rel_image_path_no_ext = join(
                person_1, '{}_{}'.format(
                    person_1, '%04d' %
                    int(image_num_1)))
            rel_image_path_1 = _add_extension(rel_image_path_no_ext, image_dir)
            is_same_person = False
        if (exists(join(image_dir, rel_image_path_0))
                and exists(join(image_dir, rel_image_path_1))):
            paths.append((rel_image_path_0, rel_image_path_1))
            labels.append(is_same_person)
        else:
            err = '{} with .jpg or .png extensions'.format(
                rel_image_path_no_ext)
            raise FileNotFoundError(err)
    return paths, labels


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


def _score_k_fold(thresholds: np.ndarray,
                  embeddings1: np.ndarray,
                  embeddings2: np.ndarray,
                  labels: np.ndarray,
                  num_folds: int,
                  distance_metric: DistanceMetric,
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
        dist = utils.embedding_distance_bulk(
            (embeddings1 - mean) / stddev,
            (embeddings2 - mean) / stddev,
            distance_metric)
        best_threshold = _calculate_best_threshold(thresholds,
                                                   dist[train_set],
                                                   labels[train_set],
                                                   threshold_metric)
        predictions = np.less(dist[test_set], best_threshold)
        accuracy[fold_idx] = accuracy_score(labels[test_set], predictions)
        recall[fold_idx] = recall_score(labels[test_set], predictions)
        precision[fold_idx] = precision_score(labels[test_set], predictions)
    return accuracy, recall, precision


def _get_target_faces(embeddings1: List[FaceVector],
                      embeddings2: List[FaceVector],
                      distance_metric: DistanceMetric,
                      is_match: bool) -> Tuple[FaceVector, FaceVector]:
    X, Y = zip(*[(emb1, emb2) for emb1 in embeddings1 for emb2 in embeddings2])
    distances = _distance_between_embeddings(X, Y, distance_metric)
    distance_criteria = min if is_match else max
    index, _ = distance_criteria(enumerate(distances), key=lambda x: x[1])
    return X[index], Y[index]


def _get_container_metrics(face_vectors: List[List[FaceVector]]) -> Tuple[
        int,
        int,
        float]:
    num_expected = len(face_vectors)
    num_missing = sum([1 for i in face_vectors if not i])
    percentage_missing = 100 * (num_missing / num_expected)
    return num_expected, num_missing, percentage_missing


def _remove_empty_embeddings(args: Namespace,
                             embeddings: np.ndarray,
                             labels: np.ndarray) -> Tuple[np.ndarray,
                                                          np.ndarray]:
    if args.remove_empty_embeddings:
        embs_filter = embeddings == np.asarray([[0] * args.embedding_size])
        empty_indices = np.where(np.all(embs_filter, axis=1))[0]
        pair_empty_indices = np.asarray([i + 1
                                         if i % 2 == 0
                                         else i - 1
                                         for i in empty_indices])
        embedding_indices = np.unique(np.concatenate((empty_indices,
                                                      pair_empty_indices)))
        embeddings = np.delete(embeddings, embedding_indices, axis=0)
        label_indices = np.unique(embedding_indices // 2)
        labels = np.delete(labels, label_indices, axis=0)
    return embeddings, labels


def _prealigned(args: Namespace, num_matches_mismatches: int,
                face_vectors: List[List[FaceVector]]) -> np.ndarray:
    if args.prealigned:
        embeddings = np.asarray([[0] * args.embedding_size
                                 if not embedding else embedding[0]
                                 for embedding in face_vectors])
    else:
        embeddings = []
        is_match = False
        for i, (embs1, embs2) in enumerate(zip(face_vectors[0::2],
                                               face_vectors[1::2])):
            if i % num_matches_mismatches == 0:
                is_match = not is_match

            embs1, embs2 = _get_target_faces(
                embs1 or [[0] * args.embedding_size],
                embs2 or [[0] * args.embedding_size],
                getattr(DistanceMetric, args.distance_metric),
                is_match)
            embeddings += [embs1, embs2]
        embeddings = np.asarray(embeddings)
    return embeddings


def _handle_flags(args: Namespace,
                  num_matches_mismatches: int,
                  face_vectors: List[List[FaceVector]],
                  labels: np.ndarray) -> Tuple[np.ndarray,
                                               np.ndarray]:
    embeddings = _prealigned(args, num_matches_mismatches, face_vectors)
    embeddings, labels = _remove_empty_embeddings(args, embeddings, labels)
    return embeddings, labels


def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        help='Path to the image directory.')
    parser.add_argument('--pairs_file_name',
                        type=str,
                        required=True,
                        help='Filename of pairs.txt')
    parser.add_argument('--face_verification_container',
                        type=str,
                        required=True,
                        help='Name of docker container for face verification')
    parser.add_argument('--num_folds',
                        type=int,
                        required=True,
                        help='Number of cross validation folds')
    distance_metrics = [
        str(metric).replace(
            '{}.'.format(
                DistanceMetric.__qualname__),
            '') for metric in DistanceMetric]
    parser.add_argument(
        '--distance_metric',
        type=str,
        required=True,
        choices=distance_metrics,
        help="Distance metric for face verification: {}.".format(distance_metrics))
    parser.add_argument('--threshold_start',
                        type=float,
                        required=True,
                        help='Start value for distance threshold.')
    parser.add_argument('--threshold_end',
                        type=float,
                        required=True,
                        help='End value for distance threshold')
    parser.add_argument('--threshold_metric',
                        type=str,
                        required=True,
                        choices=['ACCURACY', 'PRECISION', 'RECALL'],
                        help='metric for calculating threshold automatically')
    parser.add_argument(
        '--embedding_size',
        type=int,
        required=True,
        help='Size of face vectors from face_verification_container.')
    parser.add_argument(
        '--threshold_step',
        type=float,
        required=True,
        help='Step size for iterating in cross validation search.')
    parser.add_argument(
        '--remove_empty_embeddings',
        action='store_true',
        help='Instead of a default encoding for images where\
faces are not detected, remove them')
    parser.add_argument(
        '--subtract_mean',
        action='store_true',
        help="Subtract mean of embeddings before distance calculation.")
    parser.add_argument(
        '--divide_stddev',
        action='store_true',
        help="Divide embeddings by stddev before distance calculation.")
    parser.add_argument(
        '--prealigned',
        action='store_true',
        help='Specify if the images have already been aligned.')
    return parser.parse_args()


def _main(args: Namespace) -> None:
    args = _parse_arguments()
    pairs, _, num_matches_mismatches = _read_pairs(args.pairs_file_name)
    pair_paths, labels = _get_paths_and_labels(args.image_dir, pairs)
    flat_paths = [join(args.image_dir, path)
                  for pair in pair_paths for path in pair]
    # facenet = identifier.Identifier(
    #     model_path='/Users/armanrahman/models/insightface/insightface_ckpt',
    #     is_insightface=True,
    #     batch_size=128)
    facenet = identifier.Identifier(
        model_path='/Users/armanrahman/models/facenet_model.pb',
        batch_size=128)

    images = map(utils.get_image_from_path_bgr, flat_paths)

    all_vectors = facenet.vectorize_all(
        images, prealigned=args.prealigned)
    face_vectors = []
    for vectors in all_vectors:
        face_vectors.append([vector.tolist() for vector in vectors])
    num_expected, num_missing, percentage_missing = _get_container_metrics(
        face_vectors)
    print('Number of expected face vectors: {}'.format(num_expected))
    print('Number of missing face vectors: {}'.format(num_missing))
    print('Percentage missing: {}'.format(percentage_missing))

    embeddings, labels = _handle_flags(args,
                                       num_matches_mismatches,
                                       face_vectors,
                                       np.asarray(labels))
    accuracy, recall, precision = evaluate(embeddings,
                                           labels,
                                           args.num_folds,
                                           getattr(DistanceMetric,
                                                   args.distance_metric),
                                           args.threshold_metric,
                                           args.subtract_mean,
                                           args.divide_stddev,
                                           args.threshold_start,
                                           args.threshold_end,
                                           args.threshold_step)
    print('Accuracy: {}'.format(accuracy))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))


def _cli() -> None:
    args = _parse_arguments()
    _main(args)


if __name__ == '__main__':
    _cli()
