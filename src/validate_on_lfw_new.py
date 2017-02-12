"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            image_paths_placeholder = tf.get_default_graph().get_tensor_by_name("image_paths:0")
            labels_placeholder = tf.get_default_graph().get_tensor_by_name("labels:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            enqueue_op = tf.get_default_graph().get_operation_by_name("enqueue_op")
            label_batch = tf.get_default_graph().get_tensor_by_name("label_batch:0")
            
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            # Enqueue one epoch of image paths and labels
            labels_array = np.expand_dims(np.arange(0,len(paths)),1)
            image_paths_array = np.expand_dims(np.array(paths),1)
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
            
            embedding_size = embeddings.get_shape()[1]
            nrof_images = len(actual_issame)*2
            nrof_batches = nrof_images // args.lfw_batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            lab_array = np.zeros((nrof_images,))
            for _ in range(nrof_batches):
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.lfw_batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab] = emb
                
            assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
        
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            
            facenet.plot_roc(fpr, tpr, 'NN4')
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
