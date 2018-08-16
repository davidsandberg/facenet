class InsightFace(object):
    session = None
    graph = None
    output_ops = []
    input_ops = []
    feed_dict = {}

    def __init__(self, model_fp, device: str='/cpu:0', dropout_rate: float=0.1,
                 frozen=True, image_size: int=112):
        import tensorflow as tf
        self.model_fp = model_fp
        self.dropout_rate = dropout_rate
        self.input_tensor_names = ['img_inputs:0', 'dropout_rate:0']
        self.output_tensor_names = ['resnet_v1_50/E_BN2/Identity:0']
        self.frozen = frozen
        self.image_size = (image_size, image_size)

        with tf.device(device):
            self._load_graph()
            self._init_predictor()

    def _load_graph(self):
        if self.frozen:
            self._load_frozen_graph()
        else:
            self._restore_from_ckpt()

    def _restore_from_ckpt(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, self.model_fp)

    def _load_frozen_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        tf.get_default_graph()

    def _init_predictor(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config=tf_config, graph=self.graph)
            self._fetch_tensors()

    def _fetch_tensors(self):
        assert len(self.input_tensor_names) > 0
        assert len(self.output_tensor_names) > 0
        for _tensor_name in self.input_tensor_names:
            _op = self.graph.get_tensor_by_name(_tensor_name)
            self.input_ops.append(_op)
            self.feed_dict[_op] = None
        for _tensor_name in self.output_tensor_names:
            _op = self.graph.get_tensor_by_name(_tensor_name)
            self.output_ops.append(_op)

    def _set_feed_dict(self, data):
        assert len(data) == len(self.input_ops)
        with self.graph.as_default():
            for ind, op in enumerate(self.input_ops):
                self.feed_dict[op] = data[ind]

    def _clean_image(self, image: Image) -> np.ndarray:
        clean_image = cv2.resize(image, self.image_size)
        return clean_image

    def generate_embedding(self, image: Image) -> Embedding:
        clean_image = self._clean_image(image)
        input_data = [np.expand_dims(clean_image, axis=0), self.dropout_rate]
        with self.graph.as_default():
            self._set_feed_dict(data=input_data)
            embedding = self.session.run(
                self.output_ops, feed_dict=self.feed_dict)
        return np.reshape(embedding[0], -1)

    def generate_embeddings(self,
                            all_images: ImageGenerator) -> List[Embedding]:
        images = [self._clean_image(image) for image in all_images]
        input_data = [np.asarray(images), self.dropout_rate]
        if images:
            with self.graph.as_default():
                self._set_feed_dict(data=input_data)
                embeddings_array = self.session.run(
                    self.output_ops, feed_dict=self.feed_dict)[0]
                embeddings = [embedding for embedding in embeddings_array]
            return embeddings
        return []

    def get_face_embeddings(self,
                            all_faces: FacesGenerator,
                            save_memory: bool=False) -> FacesGenerator:
        """Generates embeddings from generator of Faces
        Keyword Arguments:
            save_memory -- save memory by deleting image from Face object  (default: {False})
        """
        face_list = list(all_faces)
        images = [self._clean_image(face.image)
                  for faces in face_list for face in faces]
        if len(images) > 1:
            input_data = [np.asarray(images), self.dropout_rate]
        elif len(images) == 1:
            input_data = [np.expand_dims(images[0], axis=0), self.dropout_rate]
        if face_list and images:
            with self.graph.as_default():
                self._set_feed_dict(data=input_data)
                embeddings_array = self.session.run(
                    self.output_ops, feed_dict=self.feed_dict)[0]
                index = 0
                for faces in face_list:
                    for face in faces:
                        if save_memory:
                            face.image = None
                            face.container_image = None
                        face.embedding = embeddings_array[index]
                        index += 1
                    yield faces

    @staticmethod
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

    def tear_down(self):
        self.session.close()
        tf.reset_default_graph()
        self.session = None
        self.graph = None
