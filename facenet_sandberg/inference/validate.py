import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

from facenet_sandberg import utils
from facenet_sandberg.common_types import DistanceMetric, ThresholdMetric
from facenet_sandberg.config import ValidateConfig
from facenet_sandberg.inference import Identifier

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
             threshold_metric: ThresholdMetric,
             subtract_mean: bool,
             divide_stddev: bool,
             threshold_start: float,
             threshold_end: float,
             threshold_step: float) -> Tuple[np.float, np.float, np.float]:
    import pdb; pdb.set_trace()
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


def _calculate_best_threshold(thresholds: np.ndarray,
                              dist: np.ndarray,
                              labels: np.ndarray,
                              threshold_metric: ThresholdMetric) -> np.float:
    if threshold_metric == ThresholdMetric.ACCURACY:
        threshold_score = accuracy_score
    elif threshold_metric == ThresholdMetric.PRECISION:
        threshold_score = precision_score
    elif threshold_metric == ThresholdMetric.RECALL:
        threshold_score = recall_score
    threshold_scores = np.zeros((len(thresholds)))
    for threshold_idx, threshold in enumerate(thresholds):
        import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
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
        e1 = (embeddings1 - mean) / stddev
        e2 = (embeddings2 - mean) / stddev
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
    import pdb; pdb.set_trace()
    X, Y = zip(*[(emb1, emb2) for emb1 in embeddings1 for emb2 in embeddings2])
    distances = utils.embedding_distance_bulk(X, Y, distance_metric)
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


def _remove_empty_embeddings(config: ValidateConfig,
                             embeddings: np.ndarray,
                             labels: np.ndarray) -> Tuple[np.ndarray,
                                                          np.ndarray]:
    if config.remove_empty_embeddings:
        embs_filter = embeddings == np.asarray([[0] * config.embedding_size])
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


def _prealigned(config: ValidateConfig, num_matches_mismatches: int,
                face_vectors: List[List[FaceVector]]) -> np.ndarray:
    if config.prealigned:
        embeddings = np.asarray([[0] * config.embedding_size
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
                embs1 or [[0] * config.embedding_size],
                embs2 or [[0] * config.embedding_size],
                config.distance_metric,
                is_match)
            embeddings += [embs1, embs2]
        embeddings = np.asarray(embeddings)
    return embeddings


def _handle_flags(config: ValidateConfig,
                  num_matches_mismatches: int,
                  face_vectors: List[List[FaceVector]],
                  labels: np.ndarray) -> Tuple[np.ndarray,
                                               np.ndarray]:
    embeddings = _prealigned(config, num_matches_mismatches, face_vectors)
    embeddings, labels = _remove_empty_embeddings(config, embeddings, labels)
    return embeddings, labels


def _parse_arguments(argv):
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        help='Path to validate config file',
        default='validate_config.json')
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to facial recognition model (facenet or insightface)')
    parser.add_argument(
        '--is_insightface',
        help='Set this flag if using insightface',
        action='store_true')
    return parser.parse_args(argv)


def validate(config_file: str,
             identifier: Identifier) -> Tuple[np.float, np.float, np.float]:
    config = ValidateConfig(config_file)
    pairs, _, num_matches_mismatches = utils.read_pairs_file(
        config.pairs_file_name)
    pair_paths, labels = utils.get_paths_and_labels(config.image_dir, pairs)
    flat_paths = [path for pair in pair_paths for path in pair]

    images = map(utils.get_image_from_path_rgb, flat_paths)

    all_vectors = identifier.vectorize_all(
        images, prealigned=config.prealigned)
    face_vectors = []
    for vectors in all_vectors:
        face_vectors.append([vector.tolist() for vector in vectors])
    num_expected, num_missing, percentage_missing = _get_container_metrics(
        face_vectors)
    print('Number of expected face vectors: {}'.format(num_expected))
    print('Number of missing face vectors: {}'.format(num_missing))
    print('Percentage missing: {}'.format(percentage_missing))

    embeddings, labels = _handle_flags(config,
                                       num_matches_mismatches,
                                       face_vectors,
                                       np.asarray(labels))
    accuracy, recall, precision = evaluate(embeddings,
                                           labels,
                                           config.num_folds,
                                           config.distance_metric,
                                           config.threshold_metric,
                                           config.subtract_mean,
                                           config.divide_stddev,
                                           config.threshold_start,
                                           config.threshold_end,
                                           config.threshold_step)
    print('Accuracy: {}'.format(accuracy))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))
    return accuracy, recall, precision


def _cli() -> None:
    args = _parse_arguments(sys.argv[1:])
    identifier = Identifier(
        model_path=args.model_path,
        is_insightface=args.is_insightface)
    validate(args.config_file, identifier)


if __name__ == '__main__':
    _cli()
