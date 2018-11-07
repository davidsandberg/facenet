from argparse import Namespace
from parser.container_parser import ContainerParser
from parser.face_vector_fill_parser import FaceVectorFillParser
from parser.face_vector_parser import FaceVectorParser
from parser.face_vector_remove_parser import FaceVectorRemoveParser
from parser.pair_parser import PairParser

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from calculator.distance_calculator import DistanceCalculator
from calculator.threshold_calculator import ThresholdCalculator
from metrics.metrics import EvaluationMetric, FaceVectorMetric


class Evaluator:

    def __init__(self,
                 face_vector_parser: FaceVectorParser,
                 threshold_calculator: ThresholdCalculator,
                 distance_calculator: DistanceCalculator) -> None:
        self._face_vector_parser = face_vector_parser
        self._threshold_calculator = threshold_calculator
        self._distance_calculator = distance_calculator

    @classmethod
    def create_evaluator(cls, args: Namespace) -> 'Evaluator':
        pair_parser = PairParser(args.pairs_fname, args.image_dir)
        container_parser = ContainerParser(pair_parser,
                                           args.model_path,
                                           args.is_insightface,
                                           args.prealigned_flag)
        face_vector_parser: FaceVectorParser
        if args.remove_empty_embeddings_flag:
            face_vector_parser = FaceVectorRemoveParser(container_parser,
                                                        args.distance_metric)
        else:
            face_vector_parser = FaceVectorFillParser(container_parser,
                                                      args.embedding_size,
                                                      args.distance_metric)
        threshold_calculator = ThresholdCalculator(args.distance_metric,
                                                   args.threshold_metric,
                                                   args.threshold_start,
                                                   args.threshold_end,
                                                   args.threshold_step)
        distance_calculator = DistanceCalculator(args.distance_metric)
        return cls(face_vector_parser,
                   threshold_calculator,
                   distance_calculator)

    def compute_metrics(self) -> FaceVectorMetric:
        return self._face_vector_parser.compute_metrics()

    def evaluate(self) -> EvaluationMetric:
        pairs = list(self._face_vector_parser.compute_pairs())
        threshold = self._threshold_calculator.calculate(pairs)
        dist = self._distance_calculator.calculate(pairs)
        predictions = np.less(dist, threshold)
        labels = [pair.is_match for pair in pairs]
        return EvaluationMetric(accuracy_score(labels, predictions),
                                recall_score(labels, predictions),
                                precision_score(labels, predictions))
