import os
from glob import glob, iglob
from urllib.request import urlopen

import cv2
import numpy as np
from facenet_sandberg import facenet
from facenet_sandberg.inference import utils
from facenet_sandberg.inference.common_types import *


def fit_bounding_box(max_y: int, max_x: int, x1: int,
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


def download_image(url: str, is_rgb: bool=True) -> Image:
    req = urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # BGR color space
    image = cv2.imdecode(arr, -1)
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_image_from_path_rgb(image_path: str) -> Image:
    # BGR color space
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_image_from_path_bgr(image_path: str) -> Image:
    # BGR color space
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image


def get_images_from_dir(
        directory: str, recursive: bool, is_rgb: bool=True) -> ImageGenerator:
    if recursive:
        image_paths = iglob(os.path.join(
            directory, '**', '*.*'), recursive=recursive)
    else:
        image_paths = iglob(os.path.join(directory, '*.*'))
    for image_path in image_paths:
        # BGR color space
        image = cv2.imread(image_path)
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image


def fix_image(image: Image) -> Image:
    if image.ndim < 2:
        image = image[:, :, np.newaxis]
    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]
    return image


def crop(image: Image, bb: List[float], margin: float) -> Image:
    """
    img = image from misc.imread, which should be in (H, W, C) format
    bb = pixel coordinates of bounding box: (x0, y0, x1, y1)
    margin = float from 0 to 1 for the amount of margin to add, relative to the
        bounding box dimensions (half margin added to each side)
    """

    if margin < 0:
        raise ValueError("the margin must be a value between 0 and 1")
    if margin > 1:
        raise ValueError(
            "the margin must be a value between 0 and 1 - this is a change from the existing API")

    img_height = image.shape[0]
    img_width = image.shape[1]
    x0, y0, x1, y1 = bb[:4]
    margin_height = (y1 - y0) * margin / 2
    margin_width = (x1 - x0) * margin / 2
    x0 = int(np.maximum(x0 - margin_width, 0))
    y0 = int(np.maximum(y0 - margin_height, 0))
    x1 = int(np.minimum(x1 + margin_width, img_width))
    y1 = int(np.minimum(y1 + margin_height, img_height))
    return image[y0:y1, x0:x1, :], (x0, y0, x1, y1)
