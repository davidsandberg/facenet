from parser.container_parser import ContainerParser
from parser.pair import Pair
from parser.parser_base import ParserBase
from parser.pipeline.parser_pipeline import ParserPipeline
from typing import Iterable

from metrics.metrics import FaceVectorMetric


class FaceVectorParser(ParserBase):

    def __init__(self,
                 container_parser: ContainerParser,
                 parser_pipeline: ParserPipeline,
                 distance_metric: str) -> None:
        self._container_parser = container_parser
        self._distance_metric = distance_metric
        self._parser_pipeline = parser_pipeline

    def compute_pairs(self) -> Iterable[Pair]:
        return self._parser_pipeline.execute_pipeline()

    def compute_metrics(self) -> FaceVectorMetric:
        pairs = list(self._container_parser.compute_pairs())
        num_expected = len(pairs)
        num_existing = sum(1 for pair in pairs if pair.image1 and pair.image2)
        num_missing = num_expected - num_existing
        percentage_missing = 100 * (num_missing / num_expected)
        return FaceVectorMetric(num_expected, num_missing, percentage_missing)
