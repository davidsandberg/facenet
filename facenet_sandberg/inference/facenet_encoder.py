import os
import warnings
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
from facenet_sandberg import facenet
from facenet_sandberg.inference import utils
from facenet_sandberg.inference.common_types import *
from scipy import misc

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Facenet:
    def __init__(
            self,
            model_path: str,
            image_height: int=160,
            image_width: int=160,
            batch_size: int=64):
        import tensorflow as tf
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(model_path)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

    def generate_embedding(self, image: Image) -> Embedding:
        h, w, c = image.shape
        assert h == self.image_height and w == self.image_width
        prewhiten_face = facenet.prewhiten(image)

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: [
            prewhiten_face], self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]

    def extract_batch(self, batch: np.ndarray) -> List[Embedding]:
        feed_dict = {
            self.images_placeholder: batch,
            self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return [embedding for embedding in embeddings]

    def generate_embeddings(self,
                            all_images: ImageGenerator) -> List[Embedding]:
        featurized_batches = []
        cur_batch = np.zeros(
            [self.batch_size, self.image_height, self.image_width, 3])
        clean_images = map(facenet.prewhiten, all_images)
        index = 0
        for image in clean_images:
            cur_batch[index] = image
            if index % (self.batch_size - 1) == 0 and index != 0:
                featurized_batches += self.extract_batch(cur_batch)
                cur_batch = np.zeros(
                    [self.batch_size, self.image_height, self.image_width, 3])
                index = -1
            index += 1
        # finish up remainder (get non zero rows)
        cur_batch = cur_batch[[np.any(image) for image in cur_batch]]
        if cur_batch.size > 0:
            featurized_batches += self.extract_batch(cur_batch)
        return featurized_batches

    def get_face_embeddings(self,
                            all_faces: FacesGenerator,
                            save_memory: bool=False) -> FacesGenerator:
        """Generates embeddings from generator of Faces
        Keyword Arguments:
            save_memory -- save memory by deleting image from Face object  (default: {False})
        """

        face_list = list(all_faces)
        total_num_faces = sum([1 for faces in face_list for face in faces])
        prewhitened_images = (
            facenet.prewhiten(
                face.image) for faces in face_list for face in faces)

        embed_array = self.generate_embeddings(prewhitened_images)
        total_num_embeddings = len(embed_array)
        assert total_num_embeddings == total_num_faces

        index = 0
        for faces in face_list:
            for face in faces:
                if save_memory:
                    face.image = None
                    face.container_image = None
                face.embedding = embed_array[index]
                index += 1
            yield faces

    def get_best_match(self, anchor: Face,
                       faces: List[Face], save_memory: bool=False) -> Face:
        anchor.embedding = self.generate_embedding(anchor.image)
        min_dist = float('inf')
        min_face = None
        for face in faces:
            face.embedding = self.generate_embedding(face.image)
            dist = utils.get_distance(anchor.embedding, face.embedding)
            if dist < min_dist:
                min_dist = dist
                min_face = face
        return min_face

    def tear_down(self):
        self.sess.close()
        self.sess = None
