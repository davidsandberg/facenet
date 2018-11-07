import json
from os.path import basename, dirname, join
from parser.pair import Pair
from parser.pair_parser import PairParser
from parser.parser_base import ParserBase
from typing import Dict, Iterable, List

from facenet_sandberg import (Identifier, get_image_from_path_bgr,
                              get_image_from_path_rgb)


class ContainerParser(ParserBase):

    def __init__(self,
                 pair_parser: PairParser,
                 model_path: str,
                 is_insightface: bool,
                 is_prealigned: bool) -> None:
        self._pair_parser = pair_parser
        self._model_path = model_path
        self._is_insightface = is_insightface
        self._is_prealigned = is_prealigned
        self.__face_vectors = None

    @property
    def _face_vectors(self):
        if not self.__face_vectors:
            self.__face_vectors = self._compute_face_vectors()
        return self.__face_vectors

    def compute_pairs(self) -> Iterable[Pair]:
        pairs = self._pair_parser.compute_pairs()
        return (Pair(image1, image2, pair.is_match)
                for image1, image2, pair in
                zip(self._face_vectors[0::2], self._face_vectors[1::2], pairs))

    def compute_metrics(self) -> Dict[str, float]:
        raise NotImplementedError()

    def _compute_face_vectors(self) -> List[List[List[float]]]:
        pairs = list(self._pair_parser.compute_pairs())

        identifier = Identifier(
            model_path=self._model_path,
            is_insightface=self._is_insightface)

        img_paths = [image_path
                     for pair in pairs
                     for image_path in [pair.image1, pair.image2]]
        if self._is_insightface:
            images = map(get_image_from_path_bgr, img_paths)
        else:
            images = map(get_image_from_path_rgb, img_paths)

        all_vectors = identifier.vectorize_all(
            images, prealigned=self._is_prealigned)
        np_to_list = []
        for vectors in all_vectors:
            np_to_list.append([vector.tolist() for vector in vectors])

        return np_to_list

    @staticmethod
    def _get_base_dir_for_volume_mapping(full_image_path: str) -> str:
        return dirname(dirname(full_image_path))
