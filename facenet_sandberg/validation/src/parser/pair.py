from typing import Generic
from typing import TypeVar

T = TypeVar('T')


class Pair(Generic[T]):
    def __init__(self, image1: T, image2: T, is_match: bool) -> None:
        self._image1 = image1
        self._image2 = image2
        self._is_match = is_match

    @property
    def image1(self):
        return self._image1

    @property
    def image2(self):
        return self._image2

    @property
    def is_match(self):
        return self._is_match
