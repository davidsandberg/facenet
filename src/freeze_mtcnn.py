"""Performs face alignment and stores face thumbnails in the output directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import align.detect_face
import argparse

def main(args):

    print('Freezing the PNet, RNet and ONet models')
    
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            align.detect_face.freeze_mtcnn(sess, None)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

