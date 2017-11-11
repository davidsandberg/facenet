"""Calculate the mean and standard deviation (per channel) over all images in a dataset
"""
# MIT License
# 
# Copyright (c) 2017 David Sandberg
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

import os.path
import sys
import random
import tensorflow as tf
import numpy as np
import argparse
import facenet

def main(args):
  
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    result_filename = os.path.join(os.path.expanduser(args.data_dir), 'statistics.txt')
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        
        # Get a list of image paths and their labels
        image_list, _ = facenet.get_image_paths_and_labels(train_set)
        nrof_images = len(image_list)
        assert nrof_images>0, 'The dataset should not be empty'
        
        input_queue = tf.train.string_input_producer(image_list, num_epochs=None,
                             shuffle=False, seed=None, capacity=32)
        
        
        nrof_preprocess_threads = 4
        images = []
        for _ in range(nrof_preprocess_threads):
            filename = input_queue.dequeue()
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents)
            image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
            
            #pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            image = tf.cast(image, tf.float32)
            images.append((image,))
    
        image_batch = tf.train.batch_join(images, batch_size=100, allow_smaller_final_batch=True)
        #mean = tf.reduce_mean(image_batch, reduction_indices=[0,1,2])
        m, v = tf.nn.moments(image_batch, [1,2])
        mean = tf.reduce_mean(m, 0)
        variance = tf.reduce_mean(v, 0)
        
        
        
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)

        with sess.as_default():

            # Training and validation loop
            print('Running training')
            nrof_batches = nrof_images // args.batch_size
            #nrof_batches = 20
            means = np.zeros(shape=(nrof_batches, 3), dtype=np.float32)
            variances = np.zeros(shape=(nrof_batches, 3), dtype=np.float32)
            for i in range(nrof_batches):
                means[i,:], variances[i,:] = sess.run([mean, variance])
                if (i+1)%10==0:
                    print('Batch: %5d/%5d, Mean: %s,  Variance: %s' % (i+1, nrof_batches, np.array_str(np.mean(means[:i,:],axis=0)), np.array_str(np.mean(variances[:i,:],axis=0))))
            dataset_mean = np.mean(means,axis=0)
            dataset_variance = np.mean(variances,axis=0)
            print('Final mean: %s' % np.array_str(dataset_mean))
            print('Final variance: %s' % np.array_str(dataset_variance))
            with open(result_filename, 'w') as text_file:
                print('Writing result to %s' % result_filename)
                text_file.write('Mean: %.5f, %.5f, %.5f\n' % (dataset_mean[0], dataset_mean[1], dataset_mean[2]))
                text_file.write('Variance: %.5f, %.5f, %.5f\n' % (dataset_variance[0], dataset_variance[1], dataset_variance[2]))
            
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
