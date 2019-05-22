import os
import warnings
from typing import List, Optional, cast

import numpy as np
import progressbar as pb
import tensorflow as tf
from scipy import misc

from facenet_sandberg import facenet, utils
from facenet_sandberg.common_types import (DistanceMetric, Embedding, Face,
                                           FacesGenerator, Image,
                                           ImageGenerator)

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Facenet:
    def __init__(
            self,
            model_path: str,
            image_height: int = 160,
            image_width: int = 160,
            batch_size: int = 64) -> None:
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

    def extract_batch(self, batch: np.ndarray) -> List[Embedding]:
        feed_dict = {
            self.images_placeholder: batch,
            self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return [embedding for embedding in embeddings]

    def generate_embedding(self, image: Image) -> Embedding:
        h, w, c = image.shape
        assert h == self.image_height and w == self.image_width
        prewhiten_face = utils.fixed_standardize(image)

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: [
            prewhiten_face], self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]

    def generate_embeddings(self,
                            all_images: ImageGenerator) -> List[Embedding]:
        featurized_batches = cast(List[Embedding], [])
        clean_images = np.array(list(map(utils.fixed_standardize, all_images)))

        widgets = ['Encoding:', pb.Percentage(), ' ',
                   pb.Bar(), ' ', pb.ETA(), ' ', pb.Timer()]
        timer = pb.ProgressBar(
            widgets=widgets,
            max_value=clean_images.shape[0])
        for index in range(0, clean_images.shape[0], self.batch_size):
            end_index = min(index + self.batch_size, clean_images.shape[0])
            timer.update(end_index)
            batch = clean_images[index:end_index, :]
            featurized_batches += self.extract_batch(batch)
        timer.finish()
        return featurized_batches

    def get_face_embeddings(self,
                            all_faces: FacesGenerator,
                            save_memory: bool = False) -> FacesGenerator:
        """Generates embeddings from generator of Faces
        Keyword Arguments:
            save_memory -- save memory by deleting image from Face object  (default: {False})
        """

        face_list = list(all_faces)
        total_num_faces = sum([1 for faces in face_list for face in faces])
        images = (face.image for faces in face_list for face in faces)
        embed_array = self.generate_embeddings(images)
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

    def get_best_match(
            self,
            anchor: Face,
            faces: List[Face]) -> Optional[Face]:
        anchor_image = misc.imresize(
            anchor.image,
            (self.image_height,
             self.image_width),
            interp='bilinear')
        anchor.embedding = self.generate_embedding(anchor_image)
        min_dist = float('inf')
        min_face = faces[0] if faces else None
        for face in faces:
            face_image = misc.imresize(
                face.image, (self.image_height, self.image_width), interp='bilinear')
            face.embedding = self.generate_embedding(face_image)
            dist = utils.embedding_distance(
                anchor.embedding,
                face.embedding,
                DistanceMetric.EUCLIDEAN_SQUARED)
            if dist < min_dist and dist < 1.0:
                min_dist = dist
                min_face = face
        return min_face

    def tear_down(self):
        self.sess.close()
        self.sess = None
