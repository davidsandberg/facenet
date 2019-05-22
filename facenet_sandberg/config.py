import json

from facenet_sandberg.common_types import DistanceMetric, ThresholdMetric


def get_config(config_file: str):
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    return config


class AlignConfig:
    def __init__(self, config_file: str):
        self.config = get_config(config_file)
        self.is_rgb = self.config['IS_RGB']
        self.face_crop_height = self.config['FACE_CROP_HEIGHT']
        self.face_crop_width = self.config['FACE_CROP_WIDTH']
        self.face_crop_margin = self.config['FACE_CROP_MARGIN']
        self.scale_factor = self.config['SCALE_FACTOR']
        self.steps_threshold = self.config['STEPS_THRESHOLD']
        self.detect_multiple_faces = self.config['DETECT_MULTIPLE_FACES']
        self.use_affine = self.config['USE_AFFINE']
        self.use_faceboxes = self.config['USE_FACEBOXES']
        self.num_processes = self.config['NUM_PROCESSES']
        self.facenet_model_checkpoint = self.config['FACENET_MODEL_CHECKPOINT']
        self.input_dir = self.config['INPUT_DIR']
        self.output_dir = self.config['OUTPUT_DIR']


class ValidateConfig:
    def __init__(self, config_file: str):
        self.config = get_config(config_file)
        # Path to the image directory
        self.image_dir = self.config['IMAGE_DIR']
        # Filename of pairs.txt
        self.pairs_file_name = self.config['PAIRS_FILE_NAME']
        # Number of cross validation folds
        self.num_folds = self.config['NUM_FOLDS']
        # Distance metric for face verification
        self.distance_metric = DistanceMetric.from_str(
            self.config['DISTANCE_METRIC'])
        # Start value for distance threshold.
        self.threshold_start = self.config['THRESHOLD_START']
        # End value for distance threshold
        self.threshold_end = self.config['THRESHOLD_END']
        # Step size for iterating in cross validation search
        self.threshold_step = self.config['THRESHOLD_STEP']
        # metric for calculating threshold automatically
        self.threshold_metric = ThresholdMetric.from_str(
            self.config['THRESHOLD_METRIC'])
        # Size of face vectors
        self.embedding_size = self.config['EMBEDDING_SIZE']
        # Instead of a default encoding for images where faces are not
        # detected, remove them
        self.remove_empty_embeddings = self.config['REMOVE_EMPTY_EMBEDDINGS']
        # Subtract mean of embeddings before distance calculation.
        self.subtract_mean = self.config['SUBTRACT_MEAN']
        # Divide embeddings by stddev before distance calculation.
        self.divide_stddev = self.config['DIVIDE_STDDEV']
        # Specify if the images have already been aligned.
        self.prealigned = self.config['PREALIGNED']
