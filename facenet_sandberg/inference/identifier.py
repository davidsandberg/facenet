"""Face Detection and Recognition"""

import itertools
import os
import warnings
from typing import Dict, Generator, List, Tuple

import tensorflow as tf
from facenet_sandberg import facenet
from facenet_sandberg.inference import (facenet_encoder, insightface_encoder,
                                        mtcnn_detector)
from facenet_sandberg.inference.common_types import *

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
            batch_size: int=64):
        self.detector = mtcnn_detector.Detector()
        if is_insightface:
            self.encoder = insightface_encoder.Insightface(
                model_path=model_path,
                batch_size=batch_size)
        else:
            self.encoder = facenet_encoder.Facenet(
                model_path=model_path,
                batch_size=batch_size)
        self.threshold: float = threshold

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
                          distance_metric: int=0) -> (bool,
                                                      float):
        """Compares the distance between two embeddings

        Keyword Arguments:
            distance_metric {int} -- 0 for Euclidian distance and 1 for Cosine similarity (default: {0})

        """

        distance = utils.get_distance(embedding_1, embedding_2)
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
                    distance = utils.get_distance(face_1.embedding.reshape(
                        1, -1), face_2.embedding.reshape(1, -1), distance_metric=0)
                    if distance < match.score:
                        match.score = distance
                        match.face_1 = face_1
                        match.face_2 = face_2
            if distance < self.threshold:
                match.is_match = True
        return match

    def find_all_matches(self, image_directory: str,
                         recursive: bool) -> List[Match]:
        """Finds all matches in a directory of images
        """

        all_images = utils.get_images_from_dir(image_directory, recursive)
        all_matches = []
        all_faces_lists = self.detect_encode_all(all_images)
        all_faces: Generator[Face, None, None] = (
            face for faces in all_faces_lists for face in faces)
        # Really inefficient way to check all combinations
        for face_1, face_2 in itertools.combinations(all_faces, 2):
            is_match, score = self.compare_embedding(
                face_1.embedding, face_2.embedding)
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
