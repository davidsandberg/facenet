"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import os
import sys

def main(args):
  
    network = importlib.import_module(args.model_def, 'inference')

    with tf.Graph().as_default():
      
        image_size = 160
      
        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3), name='input')

        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Build the inference graph
        logits, endpoints = network.inference(images_placeholder, 128, 1.0, 
            phase_train=False, weight_decay=0.0)

        # Split example embeddings into anchor, positive and negative and calculate triplet loss
        embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embeddings')
      

        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            
            sess.run(tf.initialize_all_variables())
            
            
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, os.path.expanduser(args.model_file))

            # Load the model
            #print('Loading model "%s"' % args.model_file)
            #facenet.load_model(args.model_file)
            
            # Get input and output tensors
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            tpr, fpr, accuracy, val, val_std, far = lfw.validate(sess, 
                paths, actual_issame, args.seed, 60, 
                images_placeholder, phase_train_placeholder, embeddings, endpoints, nrof_folds=args.lfw_nrof_folds)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            
            facenet.plot_roc(fpr, tpr, 'NN4')
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_def', type=str, 
        help='', 
        default='models.inception_resnet_v1')
    parser.add_argument('--model_file', type=str, 
        help='File containing the model parameters as well as the model metagraph (with extension ".meta")', 
        default='~/models/facenet/20160514-234418/model.ckpt-500000')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='~/datasets/lfw/lfw_realigned/')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
