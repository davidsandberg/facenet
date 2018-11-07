from abc import ABC
from abc import abstractmethod
from parser.pair import Pair
from typing import Any
from typing import Iterable


class ParserBase(ABC):

    @abstractmethod
    def compute_pairs(self) -> Iterable[Pair]:
        pass

    @abstractmethod
    def compute_metrics(self) -> Any:
        pass
