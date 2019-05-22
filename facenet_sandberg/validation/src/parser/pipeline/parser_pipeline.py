from parser.container_parser import ContainerParser
from parser.pair import Pair
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import cast

PipelineFunction = Callable[[Iterable[Pair]], Iterable[Pair]]


class ParserPipelineEmptyException(Exception):
    pass


class ParserPipeline:
    def __init__(self, container_parser: ContainerParser) -> None:
        self._container_parser = container_parser
        self._funcs: Optional[Iterable[PipelineFunction]] = None
        self.__pairs: Optional[Iterable[Pair]] = None

    @property
    def _pairs(self) -> Iterable[Pair]:
        if not self.__pairs:
            self.__pairs = self._container_parser.compute_pairs()
        return cast(Iterable[Pair], self.__pairs)

    def build(self, funcs: Iterable[Callable[[Iterable[Pair]],
                                             Iterable[Pair]]]) -> None:
        self._funcs = funcs

    def execute_pipeline(self) -> Iterable[Pair]:
        if not self._funcs:
            err = 'Pipeline must first be built before being executed'
            raise ParserPipelineEmptyException(err)
        pairs = self._pairs
        for func in self._funcs:
            pairs = func(pairs)
        return pairs
