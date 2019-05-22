import io
import json
import os
import tempfile
import warnings
from typing import List, cast

import cv2
import docker
import numpy as np
import PIL
from mtcnn.mtcnn import MTCNN

from facenet_sandberg import utils
from facenet_sandberg.common_types import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
debug = False


class Detector:
    def __init__(
            self,
            face_crop_height: int = 160,
            face_crop_width: int = 160,
            face_crop_margin: float = .4,
            detect_multiple_faces: bool = False,
            min_face_size: int = 20,
            scale_factor: float = 0.709,
            steps_threshold: List[float] = [
                0.6,
                0.7,
                0.7],
            is_rgb: bool = True,
            use_faceboxes: bool = False,
            use_affine: bool = False) -> None:
        import tensorflow as tf
        self.mtcnn = MTCNN()
        self.face_crop_height = face_crop_height
        self.face_crop_width = face_crop_width
        self.face_crop_margin = face_crop_margin
        self.detect_multiple_faces = detect_multiple_faces
        self.min_face_size = min_face_size
        self.is_rgb = is_rgb
        self.use_faceboxes = use_faceboxes
        self.use_affine = use_affine

    def bulk_find_face(self,
                       images: ImageGenerator,
                       urls: List[str] = None,
                       detect_multiple_faces: bool = False,
                       face_limit: int = 5) -> FacesGenerator:
        for index, image in enumerate(images):
            faces = self.find_faces(image, detect_multiple_faces, face_limit)
            if urls and index < len(urls):
                for face in faces:
                    face.url = urls[index]
                yield faces
            else:
                yield faces

    def find_faces(self, image: Image, detect_multiple_faces: bool = False,
                   face_limit: int = 5) -> List[Face]:
        faces = []
        results = cast(List[AlignResult], self._get_align_results(image))
        if len(results) < face_limit:
            if not detect_multiple_faces and len(results) > 1:
                img_size = np.asarray(image.shape)[0:2]
                results = utils.get_center_box(img_size, results)
            for result in results:
                face = Face()
                bb = result.bounding_box
                if bb[2] - bb[0] < self.min_face_size or bb[3] - \
                        bb[1] < self.min_face_size:
                    pass
                # preprocess changes RGB -> BGR
                processed = utils.preprocess(
                    image,
                    self.face_crop_height,
                    self.face_crop_width,
                    self.face_crop_margin,
                    bb,
                    result.landmarks,
                    self.use_affine)
                resized = cv2.resize(
                    processed, (self.face_crop_height, self.face_crop_width))
                # RGB to BGR
                if not self.is_rgb:
                    resized = resized[..., ::-1]
                face.image = resized
                faces.append(face)
        return faces

    def _get_align_results(self, image: Image) -> List[AlignResult]:
        mtcnn_results = self.mtcnn.detect_faces(image)
        img_size = np.asarray(image.shape)[0:2]
        align_results = cast(List[AlignResult], [])
        faceboxes = cast(List[List[int]], [])
        if self.use_faceboxes:
            faceboxes = self._get_faceboxes_results(image)
        if len(mtcnn_results) >= len(faceboxes):
            for result in mtcnn_results:
                bb = result['box']
                # bb[x, y, dx, dy] -> bb[x1, y1, x2, y2]
                bb = utils.fix_mtcnn_bb(
                    img_size[0], img_size[1], bb)
                align_result = AlignResult(
                    bounding_box=bb,
                    landmarks=result['keypoints'])
                align_results.append(align_result)
        else:
            for bb in faceboxes:
                # bb[y1, x1, y2, x2] -> bb[x1, y1, x2, y2]
                bb = utils.fix_faceboxes_bb(
                    img_size[0], img_size[1], bb)
                align_result = AlignResult(bounding_box=bb)
                align_results.append(align_result)
        return align_results

    @staticmethod
    def _get_faceboxes_results(image_array: np.ndarray,
                               threshold: float = 0.8) -> List[List[int]]:
        image = PIL.Image.fromarray(image_array)
        with tempfile.NamedTemporaryFile(mode="wb", dir=os.getcwd()) as image_file:
            f = io.BytesIO()
            image.save(image_file, 'png')
            image_name = os.path.basename(image_file.name)
            base_dir = os.path.dirname(image_file.name)
            command = '--image_path=/images/' + \
                image_name + ' --threshold ' + str(threshold)
            volumes = {base_dir: {'bind': '/images', 'mode': 'ro'}}
            client = docker.from_env()
            stdout = client.containers.run('arrahm/faceboxes',
                                           command,
                                           volumes=volumes)
            return json.loads(stdout.decode('utf-8').strip())['boxes']
