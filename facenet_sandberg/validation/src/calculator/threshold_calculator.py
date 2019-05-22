from parser.pair import Pair
from typing import Callable
from typing import Iterable
from typing import Union
from typing import cast

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from calculator.calculator import Calculator
from calculator.distance_calculator import DistanceCalculator
from metrics.metrics import DistanceMetric
from metrics.metrics import ThresholdMetric
from metrics.metrics import ThresholdMetricException


# pylint: disable=too-few-public-methods
class ThresholdCalculator(Calculator):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 distance_metric: Union[str, DistanceMetric],
                 threshold_metric: Union[str, ThresholdMetric],
                 threshold_start: float,
                 threshold_end: float,
                 threshold_step: float) -> None:
        if isinstance(threshold_metric, str):
            self._threshold_metric = getattr(ThresholdMetric,
                                             cast(str, threshold_metric))
        else:
            self._threshold_metric = threshold_metric
        self._distance_metric = distance_metric
        self._threshold_start = threshold_start
        self._threshold_end = threshold_end
        self._threshold_step = threshold_step

    def calculate(self, pairs: Iterable[Pair]) -> float:
        threshold_scorer = self._get_threshold_scorer()
        dist = DistanceCalculator(self._distance_metric).calculate(pairs)
        labels = [pair.is_match for pair in pairs]
        best_score = float('-inf')
        best_threshold_index = 0
        thresholds = np.arange(self._threshold_start,
                               self._threshold_end,
                               self._threshold_step)
        for i, threshold in enumerate(thresholds):
            predictions = np.less(dist, threshold)
            score = threshold_scorer(labels, predictions)
            if score > best_score:
                best_score = score
                best_threshold_index = i
        return thresholds[best_threshold_index]

    def _get_threshold_scorer(
            self) -> Callable[[np.ndarray, np.ndarray], float]:
        if self._threshold_metric == ThresholdMetric.ACCURACY:
            return accuracy_score
        if self._threshold_metric == ThresholdMetric.PRECISION:
            return precision_score
        if self._threshold_metric == ThresholdMetric.RECALL:
            return recall_score
        if self._threshold_metric == ThresholdMetric.F1:
            return f1_score
        metrics = [str(metric) for metric in ThresholdMetric]
        err = f"Undefined {ThresholdMetric.__qualname__}. \
Choose from {metrics}"
        raise ThresholdMetricException(err)
