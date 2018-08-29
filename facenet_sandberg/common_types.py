from enum import Enum
from typing import Dict, Generator, List, Tuple

import numpy as np

Landmarks = Dict[str, Tuple[int, int]]


class DistanceMetric(Enum):
    ANGULAR_DISTANCE = 1
    EUCLIDEAN_SQUARED = 0


class PersonClass:
    "Stores the paths to images for a given person"

    def __init__(self, name: str, image_paths: List[str]) -> None:
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class AlignResult:
    def __init__(
            self, bounding_box: List[int], landmarks: Landmarks=None) -> None:
        self.bounding_box = bounding_box
        self.landmarks = landmarks


class Face:
    """Class representing a single face

    Attributes:
        name {str} -- Name of person
        bounding_box {Float[]} -- box around their face in container_image
        image {Image} -- Image cropped around face
        container_image {Image} -- Original image
        embedding {Float} -- Face embedding
        matches {Matches[]} -- List of matches to the face
        url {str} -- Url where image came from
    """

    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.matches = []
        self.url = None


class Match:
    """Class representing a match between two faces

    Attributes:
        face_1 {Face} -- Face object for person 1
        face_2 {Face} -- Face object for person 2
        score {Float} -- Distance between two face embeddings
        is_match {bool} -- whether is match between faces
    """

    def __init__(self):
        self.face_1 = Face()
        self.face_2 = Face()
        self.score = float("inf")
        self.is_match = False


Image = np.ndarray
Embedding = np.ndarray
EmbeddingsGenerator = Generator[List[np.ndarray], None, None]
ImageGenerator = Generator[np.ndarray, None, None]
FaceGenerator = Generator[Face, None, None]
FacesGenerator = Generator[List[Face], None, None]
