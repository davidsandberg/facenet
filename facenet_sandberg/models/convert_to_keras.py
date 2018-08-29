import argparse
import os
import re
import sys

import numpy as np
import tensorflow as tf
from facenet_sandberg.models.keras_inception_resnet_v1 import *

re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')


def main(tf_ckpt_path, output_base_path, output_model_name):
    weights_filename = output_model_name + '_weights.h5'
    model_filename = output_model_name + '.h5'

    npy_weights_dir, weights_dir, model_dir = create_output_directories(
        output_base_path)

    extract_tensors_from_checkpoint_file(tf_ckpt_path, npy_weights_dir)
    model = InceptionResNetV1()

    print('Loading numpy weights from', npy_weights_dir)
    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(
                    os.path.join(
                        npy_weights_dir,
                        weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)

    print('Saving weights...')
    model.save_weights(os.path.join(weights_dir, weights_filename))
    print('Saving model...')
    model.save(os.path.join(model_dir, model_filename))


def create_output_directories(output_base_path):
    npy_weights_dir = os.path.join(output_base_path, 'npy_weights')
    weights_dir = os.path.join(output_base_path, 'weights')
    model_dir = os.path.join(output_base_path, 'model')
    os.makedirs(npy_weights_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return npy_weights_dir, weights_dir, model_dir


def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name
        # and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)


def parse_arguments(argv):
    """Argument parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'tf_ckpt_path',
        type=str,
        help='Path to the directory containing pretrained tensorflow checkpoints.')

    parser.add_argument(
        'output_base_path',
        type=str,
        help='Base path for the desired output directory.')

    parser.add_argument(
        'output_model_name',
        type=str,
        help='Name for the new model (do not include .h5)')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args:
        main(
            args.tf_ckpt_path,
            args.output_base_path,
            args.output_model_name)
