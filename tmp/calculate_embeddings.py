"""Calculate embeddings for a dataset and store in a .mat file.
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
import os
import sys
import time
from tensorflow.python.platform import gfile
import scipy.io as sio

def main(args):
  
    train_set = facenet.get_dataset(args.dataset_dir)
    train_set = train_set[0:10]
  
    with tf.Graph().as_default():
      
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        nrof_images = len(image_list)
        image_indices = range(nrof_images)

        image_batch, label_batch = facenet.read_and_augument_data(image_list, image_indices, args.image_size, args.batch_size, None, 
            False, False, False, nrof_preprocess_threads=4, shuffle=False)
        with tf.Session() as sess:
            tf.train.start_queue_runners(sess=sess)
            print('Loading graphdef: %s' % args.model_file)
            with gfile.FastGFile(os.path.expanduser(args.model_file),'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                embeddings = tf.import_graph_def(graph_def, input_map={'input':image_batch}, 
                    return_elements=['embeddings:0'], name='import')[0]
                
            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = nrof_images // args.batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                t = time.time()
                emb, lab = sess.run([embeddings, label_batch])
                emb_array[lab,:] = emb
                print('Batch %d in %.3f seconds' % (i, time.time()-t))

            mdict = {'image_list':image_list, 'label_list': label_list, 'embeddings':emb_array}
            sio.savemat(args.mat_file_name, mdict)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_dir', type=str,
        help='Path to the directory containing aligned dataset.')
    parser.add_argument('model_file', type=str, 
        help='The graphdef for the model to be evaluated as a protobuf (.pb) file')
    parser.add_argument('mat_file_name', type=str,
        help='The name of the mat file to store the embeddings in.')
    parser.add_argument('--image_size', type=int,
        help='Image size.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
