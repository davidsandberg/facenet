"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

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
import glob
import os
import sys
from multiprocessing import Lock, Manager, Pool, Queue, Value
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import numpy as np

from facenet_sandberg import facenet


def evaluate(embeddings, labels, nrof_folds=10,
             distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(
        labels), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2, np.asarray(
        labels), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far


def get_paths(lfw_dir, pairs):
    """Gets full paths for image pairs and labels (same person or not)

    Arguments:
        lfw_dir {str} -- Base directory of testing data
        pairs {[[str]]} -- List of pairs of form:
                            - For same person: [name, image 1 index, image 2 index]
                            - For different: [name 1, image index 1, name 2, image index 2]

    Returns:
        [(str, str)], [bool] -- list of image pair paths and labels
    """

    nrof_skipped_pairs = 0
    path_list = []
    labels = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            is_same_person = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(
                lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            is_same_person = False
        if os.path.exists(path0) and os.path.exists(
                path1):    # Only add the pair if both paths exist
            path_list += (path0, path1)
            labels.append(is_same_person)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, labels


def transform_to_lfw_format(image_directory, num_processes=1):
    """Transforms an image dataset to lfw format image names.
       Base directory should have a folder per person with the person's name.

    Arguments:
        image_directory {str} -- base directory of people folders
    """
    all_folders = os.path.join(image_directory, "*", "")
    people_folders = glob.iglob(all_folders)
    if num_processes != 1:
        process_pool = Pool(num_processes)
        process_pool.imap(rename, people_folders)
        process_pool.close()
        process_pool.join()
    else:
        for person_folder in people_folders:
            rename(person_folder)


def rename(person_folder):
    """Renames all the images in a folder in lfw format

    Arguments:
        person_folder {str} -- path to folder named after person
    """
    all_image_paths = glob.glob(os.path.join(person_folder, "*.*"))
    all_image_paths = sorted([image for image in all_image_paths if image.endswith(
        ".jpg") or image.endswith(".png")])
    person_name = os.path.basename(os.path.normpath(person_folder))
    concat_name = '_'.join(person_name.split())
    for index, image_path in enumerate(all_image_paths):
        image_name = concat_name + '_' + '%04d' % (index + 1)
        file_ext = Path(image_path).suffix
        new_image_path = os.path.join(person_folder, image_name + file_ext)
        os.rename(image_path, new_image_path)
    os.rename(person_folder, person_folder.replace(person_name, concat_name))


def add_extension(path):
    """Adds a image file extension to the path if it exists

    Arguments:
        path {str} -- base path to image file

    Raises:
        RuntimeError -- [description]

    Returns:
        str -- base path plus image file extension
    """

    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    """Reads a pairs.txt file to array. Each file line is of format:
        - If same person: "{person} {image 1 index} {image 2 index}"
        - If different: "{person 1} {image 1 index} {person 2} {image 2 index}"

    Arguments:
        pairs_filename {str} -- path to pairs.txt file

    Returns:
        np.ndarray -- numpy array of pairs
    """

    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def parse_arguments(argv):
    """Argument parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'image_directory',
        type=str,
        help='Path to the data directory containing images to fix names')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args:
        transform_to_lfw_format(args.image_directory)
