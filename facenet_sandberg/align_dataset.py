"""Performs face alignment and stores face thumbnails in the output directory."""
import argparse
import os
import sys
import warnings
from ctypes import c_int
from multiprocessing import Lock, Value
from typing import List, Optional, Tuple, cast

import cv2
import progressbar as pb
import tensorflow as tf
from pathos.multiprocessing import ProcessPool

from facenet_sandberg.common_types import Face, FacesGenerator, PersonClass
from facenet_sandberg.inference import align, facenet_encoder
from facenet_sandberg.utils import (get_dataset, get_image_from_path_rgb,
                                    transform_to_lfw_format)

# pylint: disable=no-member
from .config import AlignConfig

# pylint: disable=no-member


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

WIDGETS = ['Aligning Dataset: ', pb.Percentage(), ' ',
           pb.Bar(), ' ', pb.ETA()]
TIMER = pb.ProgressBar(widgets=WIDGETS)
NUM_SUCESSFUL = Value(c_int)  # defaults to 0
NUM_SUCESSFUL_LOCK = Lock()
NUM_UNSECESSFUL = Value(c_int)
NUM_UNSUCESSFUL_LOCK = Lock()
NUM_IMAGES_TOTAL = Value(c_int)
NUM_IMAGES_TOTAL_LOCK = Lock()


def align_dataset(config_file: str):
    """Aligns an image dataset
    """
    config = AlignConfig(config_file)
    output_dir = os.path.expanduser(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = get_dataset(config.input_dir)

    num_images = sum(len(i) for i in dataset)
    TIMER.max_value = num_images
    TIMER.start()

    num_processes = cast(int, min(config.num_processes, os.cpu_count()))
    if num_processes == -1:
        num_processes = os.cpu_count()
    if num_processes > 1:
        process_pool = ProcessPool(num_processes)
        process_pool.imap(
            align_person, zip(
                dataset, [config] * len(dataset)))
        process_pool.close()
        process_pool.join()
    else:
        for person in dataset:
            align_person((person, config))

    transform_to_lfw_format(output_dir, num_processes)

    TIMER.finish()
    print('Total number of images: %d' % int(NUM_IMAGES_TOTAL.value))
    print('Number of faces found and aligned: %d' %
          int(NUM_SUCESSFUL.value))
    print('Number of unsuccessful: %d' %
          int(NUM_UNSECESSFUL.value))


def align_person(data: Tuple[PersonClass, AlignConfig]) -> None:
    person, config = data
    output_class_dir = os.path.join(config.output_dir, person.name)
    if already_done(person, output_class_dir):
        increment_total(len(person.image_paths))
        TIMER.update(int(NUM_IMAGES_TOTAL.value))
        return None
    detector = align.Detector(
        face_crop_height=config.face_crop_height,
        face_crop_width=config.face_crop_width,
        face_crop_margin=config.face_crop_margin,
        scale_factor=config.scale_factor,
        steps_threshold=config.scale_factor,
        detect_multiple_faces=config.detect_multiple_faces,
        use_affine=config.use_affine,
        use_faceboxes=config.use_faceboxes)

    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    all_faces = gen_all_faces(person, output_class_dir, detector)
    if config.detect_multiple_faces and config.facenet_model_checkpoint and all_faces:
        encoder = None
        anchor = get_anchor(person, output_class_dir, detector)
        if anchor:
            for faces in all_faces:
                if not faces:
                    pass
                elif len(faces) > 1:
                    if not encoder:
                        encoder = facenet_encoder.Facenet(
                            model_path=config.facenet_model_checkpoint)
                    best_face = encoder.get_best_match(anchor, faces)
                    if best_face:
                        cv2.imwrite(
                            best_face.name, cv2.cvtColor(
                                best_face.image, cv2.COLOR_RGB2BGR))
                elif len(faces) == 1:
                    cv2.imwrite(
                        faces[0].name, cv2.cvtColor(
                            faces[0].image, cv2.COLOR_RGB2BGR))
        if encoder:
            encoder.tear_down()
            del encoder
    else:
        for faces in all_faces:
            if faces:
                for face in faces:
                    cv2.imwrite(
                        face.name, cv2.cvtColor(
                            face.image, cv2.COLOR_RGB2BGR))
    del detector
    TIMER.update(int(NUM_IMAGES_TOTAL.value))


def gen_all_faces(person: PersonClass, output_class_dir: str,
                  detector: align.Detector) -> FacesGenerator:
    for image_path in person.image_paths:
        increment_total()
        output_filename = get_file_name(image_path, output_class_dir)
        if not os.path.exists(output_filename):
            faces = process_image(detector, image_path, output_filename)
            if faces:
                yield faces


def already_done(person: PersonClass, output_class_dir: str):
    total = sum(os.path.exists(get_file_name(image_path, output_class_dir))
                for image_path in person.image_paths)
    return total == len(person.image_paths)


def get_anchor(person: PersonClass, output_class_dir: str,
               detector: align.Detector) -> Optional[Face]:
    first_face = None
    for image_path in person.image_paths:
        output_filename = get_file_name(image_path, output_class_dir)
        faces = process_image(detector, image_path, output_filename)
        if faces and not first_face:
            first_face = faces[0]
        if len(faces) == 1:
            return faces[0]
    return first_face


def process_image(detector: align.Detector,
                  image_path: str, output_filename: str) -> List[Face]:
    image = get_image_from_path_rgb(image_path)
    if image is not None:
        faces = detector.find_faces(image)
        if not faces:
            increment_unsucessful()
        for person in faces:
            increment_sucessful()
            filename_base, file_extension = os.path.splitext(
                output_filename)
            output_filename_n = "{}{}".format(
                filename_base, file_extension)
            person.name = output_filename_n
        return faces
    return []


def increment_sucessful(add_amount: int=1):
    with NUM_SUCESSFUL_LOCK:
        NUM_SUCESSFUL.value += add_amount


def increment_unsucessful(add_amount: int=1):
    with NUM_UNSUCESSFUL_LOCK:
        NUM_UNSECESSFUL.value += add_amount


def increment_total(add_amount: int=1):
    with NUM_IMAGES_TOTAL_LOCK:
        NUM_IMAGES_TOTAL.value += add_amount


def get_file_name(image_path: str, output_class_dir: str) -> str:
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(
        output_class_dir, filename + '.png')
    return output_filename


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=str,
        help='Path to align config file',
        default='facenet_config.json')
    return parser.parse_args(argv)


if __name__ == '__main__':
    ARGS = parse_arguments(sys.argv[1:])
    if ARGS:
        align_dataset(ARGS.config_file)
