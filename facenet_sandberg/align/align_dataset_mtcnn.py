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
from glob import iglob
from time import sleep

import cv2
import numpy as np
import progressbar as pb
import tensorflow as tf
from facenet_sandberg import face, facenet
from facenet_sandberg.align import detect_face
from mtcnn.mtcnn import MTCNN
from scipy import misc


def main(
        input_dir,
        output_dir,
        random_order,
        image_size=182,
        margin=44,
        detect_multiple_faces=False):
    """Aligns an image dataset

    Arguments:
        input_dir {str} -- Directory with unaligned images.
        output_dir {str} -- Directory with aligned face thumbnails.
        random_order {bool} -- Shuffles the order of images to enable alignment using multiple processes.

    Keyword Arguments:
        image_size {int} -- Image size (height, width) in pixels. (default: {182})
        margin {int} -- Margin for the crop around the bounding box 
                        (height, width) in pixels. (default: {44})
        detect_multiple_faces {bool} -- Detect and align multiple faces per image. 
                                        (default: {False})
    """

    widgets = ['Aligning Dataset', pb.Percentage(), ' ',
               pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
    sleep(random.random())
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(input_dir)

    print('Creating networks and loading parameters')

    detector = face.Detector(face_crop_size=image_size, face_crop_margin=margin,
                             detect_multiple_faces=detect_multiple_faces)

    # Add a random key to the filename to allow alignment using multiple
    # processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(
        output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        num_images = sum(1 for x in iglob(
            input_dir + '/**/*.*', recursive=True))
        timer = pb.ProgressBar(widgets=widgets, maxval=num_images).start()
        if random_order:
            random.shuffle(dataset)
        for datum in dataset:
            output_class_dir = os.path.join(output_dir, datum.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if random_order:
                    random.shuffle(datum.image_paths)
            for image_path in datum.image_paths:
                timer.update(nrof_images_total)
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(
                    output_class_dir, filename + '.png')
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]
                        faces = detector.find_faces(img)
                        nrof_successfully_aligned += 1
                        for index, person in enumerate(faces):
                            filename_base, file_extension = os.path.splitext(
                                output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(
                                    filename_base, index, file_extension)
                            else:
                                output_filename_n = "{}{}".format(
                                    filename_base, file_extension)
                            misc.imsave(output_filename_n, person.image)
                            text_file.write(
                                '%s %d %d %d %d\n' %
                                (output_filename_n, person.bounding_box[0],
                                 person.bounding_box[1], person.bounding_box[2],
                                 person.bounding_box[3]))
                else:
                    print('Unable to align "%s"' % image_path)
                    text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' %
          nrof_successfully_aligned)


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
            args.detect_multiple_faces)
