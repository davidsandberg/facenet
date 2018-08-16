import os
from glob import glob, iglob
from urllib.request import urlopen

import cv2
import numpy as np
from facenet_sandberg import facenet
from facenet_sandberg.inference import utils
from facenet_sandberg.inference.common_types import *


def fit_bounding_box(max_x: int, max_y: int, x1: int,
                     y1: int, dx: int, dy: int) -> List[int]:
    x2 = x1 + dx
    y2 = y1 + dy
    x1 = max(min(x1, max_x), 0)
    x2 = max(min(x2, max_x), 0)
    y1 = max(min(y1, max_y), 0)
    y2 = max(min(y2, max_y), 0)
    return [x1, y1, x2, y2]


def get_distance(embedding_1: Embedding,
                 embedding_2: Embedding,
                 distance_metric: int=0) -> float:
    """Compares the distance between two embeddings

    Keyword Arguments:
        distance_metric {int} -- 0 for Euclidian distance and 1 for Cosine similarity (default: {0})

    """

    distance = facenet.distance(embedding_1.reshape(
        1, -1), embedding_2.reshape(1, -1), distance_metric=distance_metric)[0]
    return distance


def download_image(url: str) -> Image:
    req = urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    return fix_image(image)


def get_image_from_path(image_path: str) -> Image:
    return fix_image(cv2.imread(image_path))


def get_images_from_dir(
        directory: str, recursive: bool) -> ImageGenerator:
    if recursive:
        image_paths = iglob(os.path.join(
            directory, '**', '*.*'), recursive=recursive)
    else:
        image_paths = iglob(os.path.join(directory, '*.*'))
    for image_path in image_paths:
        yield fix_image(cv2.imread(image_path))


def fix_image(image: Image):
    if image.ndim < 2:
        image = image[:, :, np.newaxis]
    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]
    return image
