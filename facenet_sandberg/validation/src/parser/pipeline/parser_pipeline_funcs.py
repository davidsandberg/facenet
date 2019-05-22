from parser.pair import Pair
from typing import Iterable

from calculator.distance_calculator import DistanceCalculator


def fill_empty(pairs: Iterable[Pair], embedding_size: int) -> Iterable[Pair]:
    empty_embedding = [[0] * embedding_size]
    return (Pair(pair.image1 or empty_embedding,
                 pair.image2 or empty_embedding,
                 pair.is_match) for pair in pairs)


def remove_empty(pairs: Iterable[Pair]) -> Iterable[Pair]:
    return (pair for pair in pairs if pair.image1 and pair.image2)


def filter_target(pairs: Iterable[Pair],
                  distance_metric: str) -> Iterable[Pair]:
    return (_compute_target(pair, distance_metric) for pair in pairs)


def _compute_target(pair: Pair, distance_metric: str) -> Pair:
    possible_pairs = [Pair(image1, image2, pair.is_match)
                      for image1 in pair.image1
                      for image2 in pair.image2]
    distance_calculator = DistanceCalculator(distance_metric)
    distances = distance_calculator.calculate(possible_pairs)
    distance_criteria = min if pair.is_match else max
    index, _ = distance_criteria(enumerate(distances), key=lambda x: x[1])
    return possible_pairs[index]
