from typing import Dict, Generator, List, Tuple

import numpy as np


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
EmbeddingsGenerator = Generator[List[Embedding], None, None]
ImageGenerator = Generator[Image, None, None]
FacesGenerator = Generator[List[Face], None, None]
Keypoints = Dict[str, Tuple[int, int]]
