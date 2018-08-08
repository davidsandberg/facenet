"""Face Detection and Recognition"""

import itertools
import os
import pickle
from glob import glob
from urllib.request import urlopen

import cv2
import numpy as np
import tensorflow as tf
from facenet_sandberg import facenet, validate_on_lfw
from facenet_sandberg.align import align_dataset_mtcnn, detect_face
from mtcnn.mtcnn import MTCNN
from scipy import misc

debug = False


class Face:
    """Class representing a single face

    Attributes:
        name {str} -- Name of person
        bounding_box {Float[]} -- box around their face in container_image
        image {cv2 image (np array)} -- Image cropped around face
        container_image {cv2 image (np array)} -- Original image
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


class Identifier:
    """Class to detect, encode, and match faces

    Arguments:
        threshold {Float} -- Distance threshold to determine matches
    """

    def __init__(self, facenet_model_checkpoint, threshold=1.10):
        self.detector = Detector()
        self.encoder = Encoder(facenet_model_checkpoint)
        self.threshold = threshold

    def download_image(self, url):
        """Downloads an image from the url as a cv2 image

        Arguments:
            url {str} -- url of image

        Returns:
            cv2 image -- image array
        """

        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        return image

    def detect_encode(self, image, face_limit=5):
        """Detects faces in an image and encodes them

        Arguments:
            image {cv2 image (np array)} -- image to find faces and encode
            face_limit {int} -- Maximum # of faces allowed in image.
                                If over limit returns empty list

        Returns:
            Face[] -- list of Face objects with embeddings attached
        """

        faces = self.detector.find_faces(image, face_limit)
        for face in faces:
            face.embedding = self.encoder.generate_embedding(face.image)
        return faces

    def detect_encode_all(self, images, urls=None, save_memory=False):
        """For a list of images finds and encodes all faces

        Arguments:
            images {List or iterable of cv2 images} -- images to encode

        Keyword Arguments:
            urls {str[]} -- Optional list of urls to attach to Face objects.
                            Should be same length as images if used. (default: {None})
            save_memory {bool} -- Saves memory by deleting image from Face objects.
                                  Should only be used if with you have some other kind
                                  of refference to the original image like a url. (default: {False})

        Returns:
            Face[] -- List of Face objects with
        """

        all_faces = self.detector.bulk_find_face(images, urls)
        all_embeddings = self.encoder.get_all_embeddings(
            all_faces, save_memory)
        return all_embeddings

    def compare_embedding(self, embedding_1, embedding_2, distance_metric=0):
        """Compares the distance between two embeddings

        Arguments:
            embedding_1 {numpy.ndarray} -- face embedding
            embedding_2 {numpy.ndarray} -- face embedding

        Keyword Arguments:
            distance_metric {int} -- 0 for Euclidian distance and 1 for Cosine similarity (default: {0})

        Returns:
            bool, float -- returns True if match and distance
        """

        distance = facenet.distance(embedding_1.reshape(
            1, -1), embedding_2.reshape(1, -1), distance_metric=distance_metric)[0]
        is_match = False
        if distance < self.threshold:
            is_match = True
        return is_match, distance

    def compare_images(self, image_1, image_2):
        """Compares two images for matching faces

        Arguments:
            image_1 {cv2 image (np array)} -- openCV image
            image_2 {cv2 image (np array)} -- openCV image

        Returns:
            Match -- Match object which has the two images, is_match, and score
        """

        match = Match()
        image_1_faces = self.detect_encode(image_1)
        image_2_faces = self.detect_encode(image_2)
        if image_1_faces and image_2_faces:
            for face_1 in image_1_faces:
                for face_2 in image_2_faces:
                    distance = facenet.distance(face_1.embedding.reshape(
                        1, -1), face_2.embedding.reshape(1, -1), distance_metric=0)[0]
                    if distance < match.score:
                        match.score = distance
                        match.face_1 = face_1
                        match.face_2 = face_2
            if distance < self.threshold:
                match.is_match = True
        return match

    def find_all_matches(self, image_directory):
        """Finds all matches in a directory of images

        Arguments:
            image_directory {str} -- directory of images

        Returns:
            Face[], Match[] -- List of face objects and list of Match objects
        """

        all_images = glob(image_directory + '/*')
        all_matches = []
        all_faces = self.detect_encode_all(all_images)
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
        return all_faces, all_matches

    def tear_down(self):
        self.encoder.tear_down()


class Encoder:
    def __init__(self, facenet_model_checkpoint):
        import tensorflow as tf
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)
        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")

    def generate_embedding(self, image):
        """Generates embeddings for a Face object with image

        Arguments:
            image {cv2 image (np array)} -- Image of face. Should be aligned.

        Returns:
            numpy.ndarray -- a single vector representing a face embedding
        """

        prewhiten_face = facenet.prewhiten(image)

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: [
            prewhiten_face], self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]

    def get_all_embeddings(self, all_faces, save_memory=False):
        """Generates embeddings for list of images

        Arguments:
            all_faces {cv2 image[]} -- array of face images

        Keyword Arguments:
            save_memory {bool} -- save memory by deleting image from Face object  (default: {False})

        Returns:
            [type] -- [description]
        """

        all_images = [facenet.prewhiten(face.image) for face in all_faces]

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: all_images,
                     self.phase_train_placeholder: False}
        embed_array = self.sess.run(self.embeddings, feed_dict=feed_dict)

        for index, face in enumerate(all_faces):
            if save_memory:
                face.image = None
            face.embedding = embed_array[index]
        return all_faces

    def tear_down(self):
        if tf.get_default_session():
            tf.get_default_session().close()


class Detector:
    # face detection parameters
    def __init__(self, face_crop_size=160, face_crop_margin=32, detect_multiple_faces=True,
                 min_face_size=20, scale_factor=0.709, steps_threshold=[0.6, 0.7, 0.7]):
        self.detector = MTCNN(weights_file=None, min_face_size=min_face_size,
                              steps_threshold=steps_threshold, scale_factor=scale_factor)
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.detect_multiple_faces = detect_multiple_faces

    def bulk_find_face(self, images, urls=None, face_limit=5):
        all_faces = []
        for index, image in enumerate(images):
            faces = self.find_faces(image, face_limit)
            if urls and index < len(urls):
                for face in faces:
                    face.url = urls[index]
                    all_faces.append(face)
            else:
                all_faces += faces
        return all_faces

    def find_faces(self, image, face_limit=5):
        if isinstance(image, str):
            image = cv2.imread(image)
        faces = []
        results = self.detector.detect_faces(image)
        img_size = np.asarray(image.shape)[0:2]
        if len(results) < face_limit:
            for result in results:
                face = Face()
                # bb[x, y, dx, dy]
                bb = result['box']
                bb[2] = bb[0] + bb[2]
                bb[3] = bb[1] + bb[3]
                bb = self._fit_bounding_box(img_size[0], img_size[1], bb[0], bb[1], bb[2], bb[3])
                cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                
                bb[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
                bb[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
                bb[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
                bb[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])

                face.bounding_box = bb
                face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

                faces.append(face)
        return faces
    
    def _fit_bounding_box(self, max_x, max_y, x1, y1, x2, y2):
        x1 = max(min(x1, max_x), 0)
        x2 = max(min(x2, max_x), 0)
        y1 = max(min(y1, max_y), 0)
        y2 = max(min(y2, max_y), 0)
        return [x1, y1, x2, y2]



def align_dataset(input_dir, output_dir, image_size=182,
                  margin=44, random_order=False, detect_multiple_faces=False):
    align_dataset_mtcnn.main(
        input_dir, output_dir, image_size, margin, random_order, detect_multiple_faces)


def test_dataset(lfw_dir, model, lfw_pairs, use_flipped_images, subtract_mean,
                 use_fixed_image_standardization, image_size=160, lfw_nrof_folds=10,
                 distance_metric=0, lfw_batch_size=128):
    validate_on_lfw.main(lfw_dir, model, lfw_pairs, use_flipped_images, subtract_mean,
                         use_fixed_image_standardization, image_size, lfw_nrof_folds,
                         distance_metric, lfw_batch_size)
