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
import scipy.io as sio
import importlib
import math

def main(args):
    network = importlib.import_module(args.model_def, 'inference')
  
    model_dir = '/media/data/DeepLearning/models/facenet/20161231-150622'
    model = os.path.join(os.path.expanduser(model_dir),'model-20161231-150622.ckpt-80000')

  
    train_set = facenet.get_dataset(args.dataset_dir)
    #train_set = train_set[0:50]
  
    with tf.Graph().as_default():
      
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        nrof_images = len(image_list)
        image_indices = range(nrof_images)

        image_batch, label_batch = facenet.read_and_augument_data(image_list, image_indices, args.image_size, args.batch_size, None, 
            False, False, False, nrof_preprocess_threads=4, shuffle=False)
        prelogits, _ = network.inference(image_batch, 1.0, 
            phase_train=False, weight_decay=0.0, reuse=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            saver.restore(sess, model)
            tf.train.start_queue_runners(sess=sess)
                
            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = int(math.ceil(nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                t = time.time()
                emb, lab = sess.run([embeddings, label_batch])
                emb_array[lab,:] = emb
                print('Batch %d in %.3f seconds' % (i, time.time()-t))

            nrof_classes = len(train_set)
            class_variance = np.zeros((nrof_classes,))
            class_center = np.zeros((nrof_classes,embedding_size))
            class_names = [cls.name for cls in train_set]
            label_array = np.array(label_list)
            for cls in set(label_list):
                idx = np.where(label_array==cls)[0]
                center = np.mean(emb_array[idx,:], axis=0)
                diffs = emb_array[idx,:] - center
                dists_sqr = np.sum(np.square(diffs), axis=1)
                class_variance[cls] = np.mean(dists_sqr)
                class_center[cls,:] = center
                
            mdict = {'class_names':class_names, 'image_list':image_list, 'label_list':label_list, 'class_variance':class_variance, 'class_center':class_center }
            sio.savemat(args.mat_file_name, mdict)
            
#             nrof_embeddings_per_matrix = 50000
#             nrof_matrices = int(math.ceil(nrof_images / nrof_embeddings_per_matrix))
#             mdict = {'image_list':image_list, 'label_list': label_list }
#             for i in range(nrof_matrices):
#                 mdict['embeddings_%05d' % i] = emb_array[(i*nrof_embeddings_per_matrix):((i+1)*nrof_embeddings_per_matrix)]
#             sio.savemat('casia_embeddings_test1.mat', mdict)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_dir', type=str,
        help='Path to the directory containing aligned dataset.')
    parser.add_argument('model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.')
    parser.add_argument('mat_file_name', type=str,
        help='The name of the mat file to store the embeddings in.')
    parser.add_argument('--image_size', type=int,
        help='Image size.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
