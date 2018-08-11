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
from facenet_sandberg import face, facenet
from pathos.multiprocessing import ProcessPool
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def main(
        input_dir: str,
        output_dir: str,
        random_order: bool=False,
        image_size: int=182,
        margin: int=44,
        detect_multiple_faces: bool=False,
        num_processes: int=1):
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
    """

    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

    dataset = facenet.get_dataset(input_dir)
    if random_order:
        random.shuffle(dataset)

    input_dir_all = os.path.join(input_dir, '**', '*.*')
    num_images = sum(1 for x in iglob(
        input_dir_all, recursive=True))

    num_processes = min(num_processes, os.cpu_count())

    aligner = Aligner(
        image_size=image_size,
        margin=margin,
        detect_multiple_faces=detect_multiple_faces,
        output_dir=output_dir,
        random_order=random_order,
        num_processes=num_processes,
        num_images=num_images)

    aligner.align_multiprocess(dataset=dataset)

    print('Creating networks and loading parameters')


class Aligner:

    def __init__(self, image_size: int, margin: int, detect_multiple_faces: bool,
                 output_dir: str, random_order: bool, num_processes: int, num_images: int):
        widgets = ['Aligning Dataset', pb.Percentage(), ' ',
                   pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        self.image_size = image_size
        self.margin = margin
        self.detect_multiple_faces = detect_multiple_faces
        self.output_dir = output_dir
        self.random_order = random_order
        self.num_processes = num_processes
        self.timer = pb.ProgressBar(widgets=widgets, maxval=num_images).start()
        self.num_sucessful = Value(c_int)  # defaults to 0
        self.num_sucessful_lock = Lock()
        self.num_images_total = Value(c_int)
        self.num_images_total_lock = Lock()

    def align_multiprocess(self, dataset: List[facenet.PersonClass]):
        if self.num_processes > 1:
            process_pool = ProcessPool(self.num_processes)
            process_pool.imap(self.align, dataset)
            process_pool.close()
            process_pool.join()
        else:
            for person in dataset:
                self.align(person)
        print('Total number of images: %d' % int(self.num_images_total.value))
        print('Number of successfully aligned images: %d' %
              int(self.num_sucessful.value))

    def align(self, person: facenet.PersonClass):
        # import pdb;pdb.set_trace()
        detector = face.Detector(
            face_crop_size=self.image_size,
            face_crop_margin=self.margin,
            detect_multiple_faces=self.detect_multiple_faces)
        # Add a random key to the filename to allow alignment using multiple
        # processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(
            self.output_dir, 'bounding_boxes_%05d.txt' % random_key)
        output_class_dir = os.path.join(self.output_dir, person.name)

        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if self.random_order:
                random.shuffle(person.image_paths)

        with open(bounding_boxes_filename, "w") as text_file:
            for image_path in person.image_paths:
                self.increment_total()
                self.process_image(detector, image_path,
                                   text_file, output_class_dir)
        self.timer.update(int(self.num_sucessful.value))

    def process_image(self, detector, image_path: str,
                      text_file: str, output_class_dir: str):
        output_filename = self.get_file_name(image_path, output_class_dir)
        if not os.path.exists(output_filename):
            try:
                image = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as error:
                error_message = '{}: {}'.format(image_path, error)
                print(error_message)
            else:
                image = self.fix_image(
                    image, image_path, output_filename, text_file)
                faces = detector.find_faces(image)
                for index, person in enumerate(faces):
                    self.increment_sucessful()
                    filename_base, file_extension = os.path.splitext(
                        output_filename)
                    if self.detect_multiple_faces:
                        output_filename_n = "{}_{}{}".format(
                            filename_base, index, file_extension)
                    else:
                        output_filename_n = "{}{}".format(
                            filename_base, file_extension)
                    misc.imsave(output_filename_n, person.image)
                    text_file.write(
                        '%s %d %d %d %d\n' %
                        (output_filename_n,
                         person.bounding_box[0],
                         person.bounding_box[1],
                         person.bounding_box[2],
                         person.bounding_box[3]))
        else:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))

    def increment_sucessful(self, add_amount: int=1):
        with self.num_sucessful_lock:
            self.num_sucessful.value += add_amount

    def increment_total(self, add_amount: int=1):
        with self.num_images_total_lock:
            self.num_images_total.value += add_amount

    @staticmethod
    def fix_image(image: np.ndarray, image_path: str,
                  output_filename: str, text_file: str):
        if image.ndim < 2:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))
        if image.ndim == 2:
            image = facenet.to_rgb(image)
        image = image[:, :, 0:3]
        return image

    @staticmethod
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
        type=bool,
        help='Detect and align multiple faces per image.',
        default=False)
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
            args.num_processes)
