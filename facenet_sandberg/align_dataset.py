"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import warnings
from ctypes import c_int
from glob import iglob
from multiprocessing import Lock, Value
from typing import List, Optional, cast

import cv2
import numpy as np
import progressbar as pb
import tensorflow as tf
from facenet_sandberg import facenet
from facenet_sandberg.common_types import *
from facenet_sandberg.inference import align, facenet_encoder
from facenet_sandberg.utils import *
from pathos.multiprocessing import ProcessPool
from scipy import misc


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

widgets = ['Aligning Dataset: ', pb.Percentage(), ' ',
           pb.Bar(), ' ', pb.ETA()]
global_image_height = 0
global_image_width = 0
global_margin = 0.0
global_scale_factor = 0.0
global_steps_threshold = [0.0]
global_detect_multiple_faces = False
global_use_faceboxes = False
global_use_affine = False
global_output_dir = ""
global_facenet_model_checkpoint = ""
timer = pb.ProgressBar(widgets=widgets)
num_sucessful = Value(c_int)  # defaults to 0
num_sucessful_lock = Lock()
num_unsucessful = Value(c_int)
num_unsucessful_lock = Lock()
num_images_total = Value(c_int)
num_images_total_lock = Lock()


def main(
        input_dir: str,
        output_dir: str,
        image_height: int = 182,
        image_width: int = 182,
        margin: float = 0.4,
        scale_factor: float = 0.0,
        steps_threshold: List[float] = [0.6, 0.7, 0.7],
        detect_multiple_faces: bool = False,
        use_faceboxes: bool = False,
        use_affine: bool = False,
        num_processes: int = 1,
        facenet_model_checkpoint: str = ''):
    """Aligns an image dataset

    Arguments:
        input_dir {str} -- Directory with unaligned images.
        output_dir {str} -- Directory with aligned face thumbnails.

    Keyword Arguments:
        image_size {int} -- Image size (height, width) in pixels. (default: {182})
        margin {int} -- Margin for the crop around the bounding box
                        (height, width) in pixels. (default: {44})
        detect_multiple_faces {bool} -- Detect and align multiple faces per image.
                                        (default: {False})
        num_processes {int} -- Number of processes to use (default: {1})
        facenet_model_checkpoint {str} -- path to facenet model if detecting mutiple faces (default: {''})
    """
    global global_image_height
    global global_image_width
    global global_margin
    global global_scale_factor
    global global_steps_threshold
    global global_detect_multiple_faces
    global global_use_faceboxes
    global global_use_affine
    global global_output_dir
    global global_facenet_model_checkpoint
    global timer
    global_image_height = image_height
    global_image_width = image_width
    global_margin = margin
    global_scale_factor = scale_factor
    global_steps_threshold = steps_threshold
    global_detect_multiple_faces = detect_multiple_faces
    global_use_faceboxes = use_faceboxes
    global_use_affine = use_affine
    global_output_dir = output_dir
    global_facenet_model_checkpoint = facenet_model_checkpoint

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = get_dataset(input_dir)

    num_images = sum(len(i) for i in dataset)
    timer.max_value = num_images
    timer.start()

    num_processes = cast(int, min(num_processes, os.cpu_count()))
    if num_processes == -1:
        num_processes = os.cpu_count()
    if num_processes > 1:
        process_pool = ProcessPool(num_processes)
        process_pool.imap(align_person, dataset)
        process_pool.close()
        process_pool.join()
    else:
        for person in dataset:
            align_person(person)

    timer.finish()
    print('Total number of images: %d' % int(num_images_total.value))
    print('Number of faces found and aligned: %d' %
          int(num_sucessful.value))
    print('Number of unsuccessful: %d' %
          int(num_unsucessful.value))


def align_person(person: PersonClass) -> None:
    import pdb
    pdb.set_trace()
    output_class_dir = os.path.join(global_output_dir, person.name)
    if already_done(person, output_class_dir):
        increment_total(len(person.image_paths))
        timer.update(int(num_images_total.value))
        return None
    detector = align.Detector(
        face_crop_height=global_image_height,
        face_crop_width=global_image_width,
        face_crop_margin=global_margin,
        scale_factor=global_scale_factor,
        steps_threshold=global_steps_threshold,
        detect_multiple_faces=global_detect_multiple_faces,
        use_affine=global_use_affine,
        use_faceboxes=global_use_faceboxes)

    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    all_faces = gen_all_faces(person, output_class_dir, detector)

    if global_detect_multiple_faces and global_facenet_model_checkpoint and all_faces:
        encoder = None
        anchor = get_anchor(person, output_class_dir, detector)
        if anchor:
            for faces in all_faces:
                if not faces:
                    pass
                elif len(faces) > 1:
                    if not encoder:
                        encoder = facenet_encoder.Facenet(
                            model_path=global_facenet_model_checkpoint)
                    best_face = encoder.get_best_match(anchor, faces)
                    if best_face:
                        cv2.imwrite(best_face.name, best_face.image)
                elif len(faces) == 1:
                    cv2.imwrite(faces[0].name, faces[0].image)
        if encoder:
            encoder.tear_down()
            del encoder
    else:
        for faces in all_faces:
            if faces:
                for face in faces:
                    cv2.imwrite(face.name, face.image)
    del detector
    timer.update(int(num_images_total.value))


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
        for index, person in enumerate(faces):
            increment_sucessful()
            filename_base, file_extension = os.path.splitext(
                output_filename)
            output_filename_n = "{}{}".format(
                filename_base, file_extension)
            person.name = output_filename_n
        return faces
    return []


def increment_sucessful(add_amount: int = 1):
    with num_sucessful_lock:
        num_sucessful.value += add_amount


def increment_unsucessful(add_amount: int = 1):
    with num_unsucessful_lock:
        num_unsucessful.value += add_amount


def increment_total(add_amount: int = 1):
    with num_images_total_lock:
        num_images_total.value += add_amount


def get_file_name(image_path: str, output_class_dir: str) -> str:
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(
        output_class_dir, filename + '.png')
    return output_filename


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('--facenet_model_checkpoint', type=str,
                        help='Path to facenet model', default='')
    parser.add_argument(
        '--image_height',
        type=int,
        help='Image height in pixels.',
        default=160)
    parser.add_argument(
        '--image_width',
        type=int,
        help='Image width in pixels.',
        default=160)
    parser.add_argument(
        '--margin',
        type=float,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=0.25)
    parser.add_argument(
        '--scale_factor',
        type=float,
        help='Factor to scale',
        default=0.85)
    parser.add_argument(
        '--steps_threshold',
        type=List[float],
        help='Thresholds',
        default=[0.6, 0.7, 0.7])
    parser.add_argument(
        '--detect_multiple_faces',
        help='Detect and align multiple faces per image.',
        action='store_true')
    parser.add_argument(
        '--use_faceboxes',
        help='Use Faceboxes in addition to MTCNN',
        action='store_true')
    parser.add_argument(
        '--use_affine',
        help='Affine transform image.',
        action='store_true')
    parser.add_argument(
        '--num_processes',
        type=int,
        help='Number of processes to use',
        default=1)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args:
        main(
            args.input_dir,
            args.output_dir,
            args.image_height,
            args.image_width,
            args.margin,
            args.scale_factor,
            args.steps_threshold,
            args.detect_multiple_faces,
            args.use_faceboxes,
            args.use_affine,
            args.num_processes,
            args.facenet_model_checkpoint)
