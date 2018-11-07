import math
from parser.pair import Pair
from typing import Iterable
from typing import Union
from typing import cast

import numpy as np
from sklearn.metrics.pairwise import paired_distances

from calculator.calculator import Calculator
from metrics.metrics import DistanceMetric
from metrics.metrics import DistanceMetricException


# pylint: disable=too-few-public-methods
class DistanceCalculator(Calculator):

    def __init__(self, distance_metric: Union[str, DistanceMetric]) -> None:
        if isinstance(distance_metric, str):
            self._distance_metric = getattr(DistanceMetric,
                                            cast(str, distance_metric))
        else:
            self._distance_metric = distance_metric

    def calculate(self, pairs: Iterable[Pair]) -> np.ndarray:
        embeddings1 = []
        embeddings2 = []
        for pair in pairs:
            embeddings1.append(pair.image1)
            embeddings2.append(pair.image2)
        if self._distance_metric == DistanceMetric.EUCLIDEAN_SQUARED:
            return np.square(
                paired_distances(
                    embeddings1,
                    embeddings2,
                    metric='euclidean'))
        if self._distance_metric == DistanceMetric.ANGULAR_DISTANCE:
            # Angular Distance: https://en.wikipedia.org/wiki/Cosine_similarity
            similarity = 1 - paired_distances(
                embeddings1,
                embeddings2,
                metric='cosine')
            return np.arccos(similarity) / math.pi
        metrics = [str(metric) for metric in DistanceMetric]
        err = f"Undefined {DistanceMetric.__qualname__}. \
Choose from {metrics}"
        raise DistanceMetricException(err)
