import os
import warnings
from typing import List, cast

import numpy as np
import progressbar as pb
import tensorflow as tf
import tensorlayer as tl

from facenet_sandberg import utils
from facenet_sandberg.common_types import *
from facenet_sandberg.models.L_Resnet_E_IR_fix_issue9 import get_resnet

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class Insightface:
    def __init__(self, model_path: str,
                 image_height: int = 112,
                 image_width: int = 112,
                 batch_size: int = 64) -> None:
        import tensorflow as tf
        import tensorflow.contrib.slim as slim
        from tensorflow.core.protobuf import config_pb2
        # save context
        sess, embedding_tensor, feed_dict, input_placeholder = self._get_extractor(
            model_path, image_height, image_width)
        self.embedding_tensor = embedding_tensor
        self.batch_size = batch_size
        self.sess = sess
        self.feed_dict = feed_dict
        self.input_placeholder = input_placeholder
        self.model_path = model_path
        self.image_height = image_height
        self.image_width = image_width

    def _get_extractor(self, model_path: str,
                       image_height: int, image_width: int):
        images = tf.placeholder(
            name='img_inputs',
            shape=[
                None,
                image_height,
                image_width,
                3],
            dtype=tf.float32)
        dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        with tl.ops.suppress_stdout():
            tl.layers.set_name_reuse(True)
            test_net = get_resnet(
                images,
                50,
                type='ir',
                w_init=w_init_method,
                trainable=False,
                reuse=tf.AUTO_REUSE,
                keep_rate=dropout_rate)
        embedding_tensor = test_net.outputs
        # define sess
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True

        sess = tf.Session(config=gpu_config)
        # init all variables
        sess.run(tf.global_variables_initializer())

        # restore weights
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        feed_dict = {images: None, dropout_rate: 1.0}
        return sess, embedding_tensor, feed_dict, images

    def extract_batch(self, batch: np.ndarray) -> List[Embedding]:
        self.feed_dict.setdefault(self.input_placeholder, None)
        self.feed_dict[self.input_placeholder] = batch
        feat = self.sess.run([self.embedding_tensor], feed_dict=self.feed_dict)
        feat = np.array(feat)
        if feat.ndim > 2:
            feat = np.squeeze(feat, axis=0)
        return [embedding for embedding in feat]

    def generate_embedding(self, image: Image) -> Embedding:
        image = utils.fixed_standardize(image)
        self.feed_dict.setdefault(self.input_placeholder, None)
        self.feed_dict[self.input_placeholder] = [image]
        feat = self.sess.run([self.embedding_tensor], feed_dict=self.feed_dict)
        feat = np.array(feat)
        feat = np.squeeze(feat)
        return feat

    def generate_embeddings(
            self,
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

    def tear_down(self):
        self.sess.close()
