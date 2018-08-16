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
import random
import sys
from ctypes import c_int
from glob import iglob
from multiprocessing import Lock, Value
from typing import List

import numpy as np
import progressbar as pb
import tensorflow as tf
from facenet_sandberg import facenet
from facenet_sandberg.inference import facenet_encoder, mtcnn_detector
from pathos.multiprocessing import ProcessPool
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

widgets = ['Aligning Dataset', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
global_image_size = None
global_margin = None
global_detect_multiple_faces = None
global_output_dir = None
global_random_order = None
global_facenet_model_checkpoint = None
timer = None
num_sucessful = Value(c_int)  # defaults to 0
num_sucessful_lock = Lock()
num_images_total = Value(c_int)
num_images_total_lock = Lock()


def main(
        input_dir: str,
        output_dir: str,
        random_order: bool=False,
        image_size: int=182,
        margin: int=44,
        detect_multiple_faces: bool=False,
        num_processes: int=1,
        facenet_model_checkpoint: str=''):
    """Aligns an image dataset

    Arguments:
        input_dir {str} -- Directory with unaligned images.
        output_dir {str} -- Directory with aligned face thumbnails.

    Keyword Arguments:
        random_order {bool} -- Shuffles the order of images to enable alignment
                                using multiple processes. (default: {False})
        image_size {int} -- Image size (height, width) in pixels. (default: {182})
        margin {int} -- Margin for the crop around the bounding box
                        (height, width) in pixels. (default: {44})
        detect_multiple_faces {bool} -- Detect and align multiple faces per image.
                                        (default: {False})
        num_processes {int} -- Number of processes to use (default: {1})
        facenet_model_checkpoint {str} -- path to facenet model if detecting mutiple faces (default: {''})
    """
    global timer
    global global_image_size
    global global_margin
    global global_detect_multiple_faces
    global global_output_dir
    global global_random_order
    global global_facenet_model_checkpoint
    global_image_size = image_size
    global_margin = margin
    global_detect_multiple_faces = detect_multiple_faces
    global_output_dir = output_dir
    global_random_order = random_order
    global_facenet_model_checkpoint = facenet_model_checkpoint

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = facenet.get_dataset(input_dir)
    if random_order:
        random.shuffle(dataset)

    num_images = sum(len(i) for i in dataset)
    timer = pb.ProgressBar(widgets=widgets, maxval=num_images).start()

    num_processes = min(num_processes, os.cpu_count())
    if num_processes > 1:
        process_pool = ProcessPool(num_processes)
        process_pool.map(align, dataset)
        process_pool.close()
        process_pool.join()
    else:
        for person in dataset:
            align(person)

    timer.finish()
    print('Total number of images: %d' % int(num_images_total.value))
    print('Number of faces found and aligned: %d' %
          int(num_sucessful.value))


def align(person: facenet.PersonClass):
    detector = mtcnn_detector.Detector(
        detect_multiple_faces=global_detect_multiple_faces)
    output_class_dir = os.path.join(global_output_dir, person.name)

    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
        if global_random_order:
            random.shuffle(person.image_paths)

    all_faces = []
    for image_path in person.image_paths:
        increment_total()
        output_filename = get_file_name(image_path, output_class_dir)
        if not os.path.exists(output_filename):
            faces = process_image(detector, image_path, output_filename)
            if faces:
                all_faces.append(faces)

    if global_detect_multiple_faces and global_facenet_model_checkpoint and all_faces:
        encoder = facenet_encoder.Facenet(global_facenet_model_checkpoint)
        anchor = get_anchor(all_faces)
        if anchor:
            final_face_paths = []
            for faces in all_faces:
                if not faces:
                    pass
                if len(faces) > 1:
                    best_face = encoder.get_best_match(anchor, faces)
                    misc.imsave(best_face.name, best_face.image)
                elif len(faces) == 1:
                    misc.imsave(faces[0].name, faces[0].image)
    else:
        for faces in all_faces:
            if faces:
                for person in faces:
                    misc.imsave(person.name, person.image)
    timer.update(int(num_images_total.value))


def get_anchor(all_faces):
    for faces in all_faces:
        if faces and len(faces) == 1:
            return faces[0]
    if all_faces:
        if all_faces[0]:
            if all_faces[0][0]:
                return all_faces[0][0]
    return None


def process_image(detector, image_path: str, output_filename: str):
    if not os.path.exists(output_filename):
        try:
            image = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as error:
            # error_message = '{}: {}'.format(image_path, error)
            # print(error_message)
            return []
        else:
            image = fix_image(image, image_path)
            faces = detector.find_faces(image)
            for index, person in enumerate(faces):
                increment_sucessful()
                filename_base, file_extension = os.path.splitext(
                    output_filename)
                output_filename_n = "{}{}".format(
                    filename_base, file_extension)
                person.name = output_filename_n
            return faces
    else:
        print('Unable to align "%s"' % image_path)


def increment_sucessful(add_amount: int=1):
    with num_sucessful_lock:
        num_sucessful.value += add_amount


def increment_total(add_amount: int=1):
    with num_images_total_lock:
        num_images_total.value += add_amount


def fix_image(image: np.ndarray, image_path: str):
    if image.ndim < 2:
        print('Unable to align "%s"' % image_path)
    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]
    return image


def get_file_name(image_path: str, output_class_dir: str) -> str:
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(
        output_class_dir, filename + '.png')
    return output_filename


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('facenet_model_checkpoint', type=str,
                        help='Path to facenet model', default='')
    parser.add_argument(
        '--image_size',
        type=int,
        help='Image size (height, width) in pixels.',
        default=182)
    parser.add_argument(
        '--margin',
        type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=44)
    parser.add_argument(
        '--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.',
        action='store_true')
    parser.add_argument(
        '--detect_multiple_faces',
        help='Detect and align multiple faces per image.',
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
            args.random_order,
            args.image_size,
            args.margin,
            args.detect_multiple_faces,
            args.num_processes,
            args.facenet_model_checkpoint)
