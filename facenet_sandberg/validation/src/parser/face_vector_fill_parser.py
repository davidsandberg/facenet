from functools import partial
from parser.container_parser import ContainerParser
from parser.face_vector_parser import FaceVectorParser
from parser.pipeline.parser_pipeline import ParserPipeline
from parser.pipeline.parser_pipeline_funcs import fill_empty
from parser.pipeline.parser_pipeline_funcs import filter_target


class FaceVectorFillParser(FaceVectorParser):
    def __init__(self,
                 container_parser: ContainerParser,
                 embedding_size: int,
                 distance_metric: str) -> None:
        self._parser_pipeline = ParserPipeline(container_parser)
        self._build_pipeline(embedding_size, distance_metric)
        super().__init__(container_parser,
                         self._parser_pipeline,
                         distance_metric)

    def _build_pipeline(self,
                        embedding_size: int,
                        distance_metric: str) -> None:
        partial_fill = partial(fill_empty, embedding_size=embedding_size)
        partial_filter = partial(filter_target,
                                 distance_metric=distance_metric)
        self._parser_pipeline.build([partial_fill, partial_filter])
