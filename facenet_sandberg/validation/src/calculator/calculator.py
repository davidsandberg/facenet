from abc import ABC
from abc import abstractmethod
from parser.pair import Pair
from typing import Generic
from typing import Iterable
from typing import TypeVar

T = TypeVar('T')


class Calculator(ABC, Generic[T]):  # pylint: disable=too-few-public-methods

    @abstractmethod
    def calculate(self, pairs: Iterable[Pair]) -> T:
        pass
