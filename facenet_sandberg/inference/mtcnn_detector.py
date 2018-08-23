import os
import time
import warnings
from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np
from facenet_sandberg import facenet
from facenet_sandberg.inference import utils
from facenet_sandberg.inference.common_types import *
from mtcnn.mtcnn import MTCNN
from scipy import misc

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
debug = False


class Detector:
    def __init__(
            self,
            face_crop_height: int=160,
            face_crop_width: int=160,
            face_crop_margin: float=.4,
            detect_multiple_faces: bool=True,
            min_face_size: int=20,
            scale_factor: float=0.709,
            steps_threshold: List[float]=[
                0.6,
                0.7,
                0.7],
            is_rgb: bool=True):
        import tensorflow as tf
        self.detector = MTCNN(
            weights_file=None,
            min_face_size=min_face_size,
            steps_threshold=steps_threshold,
            scale_factor=scale_factor)
        self.face_crop_height = face_crop_height
        self.face_crop_width = face_crop_width
        self.face_crop_margin = face_crop_margin
        self.detect_multiple_faces = detect_multiple_faces
        self.min_face_size = min_face_size
        self.is_rgb = is_rgb

    def bulk_find_face(self,
                       images: ImageGenerator,
                       urls: List[str] = None,
                       detect_multiple_faces: bool=True,
                       face_limit: int=5) -> FacesGenerator:
        for index, image in enumerate(images):
            faces = self.find_faces(image, detect_multiple_faces, face_limit)
            if urls and index < len(urls):
                for face in faces:
                    face.url = urls[index]
                yield faces
            else:
                yield faces

    def find_faces(self, image: Image, detect_multiple_faces: bool=True,
                   face_limit: int=5) -> List[Face]:
        faces = []
        results = self.detector.detect_faces(image)
        img_size = np.asarray(image.shape)[0:2]
        if len(results) < face_limit:
            if not detect_multiple_faces:
                results = results[:1]
            for result in results:
                face = Face()
                bb = result['box']
                # bb[x, y, dx, dy] -> bb[x1, y1, x2, y2]
                bb = utils.fit_bounding_box(
                    img_size[0], img_size[1], bb[0], bb[1], bb[2], bb[3])
                if bb[2] - bb[0] < self.min_face_size or bb[3] - \
                        bb[1] < self.min_face_size:
                    pass
                processed = utils.preprocess(
                    image,
                    self.face_crop_height,
                    self.face_crop_width,
                    self.face_crop_margin,
                    bb,
                    result['keypoints'])
                resized = cv2.resize(
                    processed, (self.face_crop_height, self.face_crop_width))
                # BGR to RGB
                if self.is_rgb:
                    resized = resized[..., ::-1]
                face.image = resized
                faces.append(face)
        return faces
