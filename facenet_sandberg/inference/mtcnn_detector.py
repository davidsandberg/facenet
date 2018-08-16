import os
import warnings
from typing import Dict, Generator, List, Tuple

import numpy as np
from facenet_sandberg import facenet
from facenet_sandberg.inference import utils
from facenet_sandberg.inference.common_types import *
from mtcnn.mtcnn import MTCNN
from scipy import misc

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Detector:
    def __init__(
        self,
        face_crop_size: int=160,
        face_crop_margin: int=32,
        detect_multiple_faces: bool=True,
        min_face_size: int=20,
        scale_factor: float=0.709,
        steps_threshold: List[float]=[
            0.6,
            0.7,
            0.7]):
        self.detector = MTCNN(
            weights_file=None,
            min_face_size=min_face_size,
            steps_threshold=steps_threshold,
            scale_factor=scale_factor)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.detect_multiple_faces = detect_multiple_faces

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
                # bb[x, y, dx, dy]
                bb = result['box']
                bb = utils.fit_bounding_box(
                    img_size[0], img_size[1], bb[0], bb[1], bb[2], bb[3])
                cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]

                bb[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
                bb[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
                bb[2] = np.minimum(
                    bb[2] + self.face_crop_margin / 2, img_size[1])
                bb[3] = np.minimum(
                    bb[3] + self.face_crop_margin / 2, img_size[0])

                face.bounding_box = bb
                face.image = misc.imresize(
                    cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

                faces.append(face)
        return faces
