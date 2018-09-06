"""Face Detection and Recognition"""

import itertools
import os
import warnings
from typing import Dict, Generator, List, Tuple

import tensorflow as tf
from facenet_sandberg import facenet, utils
from facenet_sandberg.common_types import *
from facenet_sandberg.inference import (align, facenet_encoder,
                                        insightface_encoder)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Identifier:
    """Class to detect, encode, and match faces

    Arguments:
        threshold {Float} -- Distance threshold to determine matches
    """

    def __init__(
            self,
            model_path: str,
            threshold: float = 1.10,
            is_insightface: bool=False,
            is_centerface: bool=False,
            batch_size: int=64):
        if is_insightface:
            self.detector = mtcnn_detector.Detector(
                face_crop_height=112,
                face_crop_width=112,
                face_crop_margin=44,
                steps_threshold=[
                    0.6,
                    0.7,
                    0.9],
                scale_factor=0.85,
                is_rgb=False)
            self.encoder = insightface_encoder.Insightface(
                model_path=model_path,
                batch_size=batch_size)
        elif is_centerface:
            self.detector = mtcnn_detector.Detector(
                face_crop_height=112,
                face_crop_width=96,
                face_crop_margin=44,
                steps_threshold=[
                    0.6,
                    0.7,
                    0.9],
                scale_factor=0.85,
                is_rgb=False)
            self.encoder = insightface_encoder.Insightface(
                model_path=model_path,
                batch_size=batch_size,
                image_height=112,
                image_width=96)
        else:
            self.detector = mtcnn_detector.Detector()
            self.encoder = facenet_encoder.Facenet(
                model_path=model_path,
                batch_size=batch_size)
        self.threshold = threshold

    def vectorize(self, image: Image,
                  prealigned: bool = False,
                  detect_multiple_faces: bool=True,
                  face_limit: int = 5) -> List[Embedding]:
        """Gets face embeddings in a single image
        Keyword Arguments:
            prealigned {bool} -- is the image already aligned
            face_limit {int} -- max number of faces allowed
                                before image is discarded. (default: {5})
        """
        if not prealigned:
            faces = self.detect_encode(
                image, detect_multiple_faces, face_limit)
            vectors = [face.embedding for face in faces]
        else:
            vectors = [self.encoder.generate_embedding(image)]
        return vectors

    def vectorize_all(self,
                      images: ImageGenerator,
                      prealigned: bool = False,
                      detect_multiple_faces: bool=True,
                      face_limit: int = 5) -> List[List[Embedding]]:
        """Gets face embeddings from a generator of images
        Keyword Arguments:
            prealigned {bool} -- is the image already aligned
            face_limit {int} -- max number of faces allowed
                                before image is discarded. (default: {5})
        """
        vectors = []
        if not prealigned:
            all_faces = self.detect_encode_all(
                images=images,
                save_memory=True,
                detect_multiple_faces=detect_multiple_faces,
                face_limit=face_limit)
            for faces in all_faces:
                vectors.append([face.embedding for face in faces])
        else:
            embeddings = self.encoder.generate_embeddings(
                images)
            for embedding in embeddings:
                vectors.append([embedding])
        return vectors

    def detect_encode(self, image: Image,
                      detect_multiple_faces: bool=True,
                      face_limit: int=5) -> List[Face]:
        """Detects faces in an image and encodes them
        """

        faces = self.detector.find_faces(
            image, detect_multiple_faces, face_limit)
        for face in faces:
            face.embedding = self.encoder.generate_embedding(face.image)
        return faces

    def detect_encode_all(self,
                          images: ImageGenerator,
                          urls: [str]=None,
                          save_memory: bool=False,
                          detect_multiple_faces: bool=True,
                          face_limit: int=5) -> FacesGenerator:
        """For a list of images finds and encodes all faces

        Keyword Arguments:
            save_memory {bool} -- Saves memory by deleting image from Face objects.
                                Should only be used if with you have some other kind
                                of refference to the original image like a url. (default: {False})
        """

        all_faces = self.detector.bulk_find_face(
            images, urls, detect_multiple_faces, face_limit)
        return self.encoder.get_face_embeddings(all_faces, save_memory)

    def compare_embedding(self,
                          embedding_1: Embedding,
                          embedding_2: Embedding,
                          distance_metric: DistanceMetric) -> (bool,
                                                               float):
        """Compares the distance between two embeddings
        """

        distance = utils.embedding_distance(
            embedding_1, embedding_2, distance_metric)
        is_match = False
        if distance < self.threshold:
            is_match = True
        return is_match, distance

    def compare_images(
            self,
            image_1: Image,
            image_2: Image,
            detect_multiple_faces: bool=True,
            face_limit: int=5) -> Match:
        match = Match()
        image_1_faces = self.detect_encode(
            image_1, detect_multiple_faces, face_limit)
        image_2_faces = self.detect_encode(
            image_2, detect_multiple_faces, face_limit)
        if image_1_faces and image_2_faces:
            for face_1 in image_1_faces:
                for face_2 in image_2_faces:
                    is_match, score = self.compare_embedding(
                        face_1.embedding, face_2.embedding, distance_metric)
                    if score < match.score:
                        match.score = score
                        match.face_1 = face_1
                        match.face_2 = face_2
                        match.is_match = is_match
        return match

    def find_all_matches(self, image_directory: str, recursive: bool,
                         distance_metric: DistanceMetric=DistanceMetric.EUCLIDEAN_SQUARED) -> List[Match]:
        """Finds all matches in a directory of images
        """

        all_images = utils.get_images_from_dir(image_directory, recursive)
        all_matches = []
        all_faces_lists = self.detect_encode_all(all_images)
        all_faces = (face for faces in all_faces_lists for face in faces)
        # Really inefficient way to check all combinations
        for face_1, face_2 in itertools.combinations(all_faces, 2):
            is_match, score = self.compare_embedding(
                face_1.embedding, face_2.embedding, distance_metric)
            if is_match:
                match = Match()
                match.face_1 = face_1
                match.face_2 = face_2
                match.is_match = True
                match.score = score
                all_matches.append(match)
                face_1.matches.append(match)
                face_2.matches.append(match)
        return all_matches

    def tear_down(self):
        self.encoder.tear_down()
