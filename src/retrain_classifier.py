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
import sys
import math
import time
import tensorflow.contrib.slim as slim

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            tf.set_random_seed(args.seed)
            global_step = tf.Variable(0, trainable=False)
            
            if args.use_split_dataset:
                dataset = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
            else:
                train_set = facenet.get_dataset(args.data_dir)
                test_set = facenet.get_dataset(args.test_data_dir)
                 
            train_images, train_labels = facenet.get_image_paths_and_labels(train_set)
            test_images, test_labels = facenet.get_image_paths_and_labels(test_set)
            nrof_classes = len(train_set)
            assert len(test_set)==nrof_classes
            
            print('Number of classes: %d' % len(train_set))
            print('Number of train images: %d' % len(train_images))
            print('Number of test images: %d' % len(test_images))
            
            # Load the model
            facenet.load_model(args.model_dir)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            for phase, paths in [['train', train_images], ['test', test_images]]:
                print('Runnning forward pass on %s images' % phase)
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i*args.batch_size
                    end_index = min((i+1)*args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
                embeddings_filename = 'embeddings_%s.npy' % phase
                print('Saving %s' % embeddings_filename)
                np.save(embeddings_filename, emb_array)
                    
            emb_array_train = np.load('embeddings_train.npy')
            emb_array_test = np.load('embeddings_test.npy')
            labels = tf.placeholder(tf.int32, [None], 'labels')
            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            logits = slim.fully_connected(embeddings, nrof_classes, activation_fn=None, 
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                    weights_regularizer=slim.l2_regularizer(args.weight_decay),
                    scope='Logits', reuse=False)
            
            nrof_batches_per_epoch_train = int(math.ceil(1.0*len(train_labels) / args.batch_size))
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                args.learning_rate_decay_epochs*nrof_batches_per_epoch_train, args.learning_rate_decay_factor, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
    
            # Calculate the average cross entropy loss across the batch
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            predictions = tf.nn.softmax(logits, name='predictions')
            
            # Calculate the total losses
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    
            # Build a Graph that trains the model with one batch of examples and updates the model parameters
            train_op = facenet.train(total_loss, global_step, args.optimizer, 
                learning_rate, args.moving_average_decay, tf.global_variables(), False)
            
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            validate_every_n_epochs = 10
            for epoch in range(args.max_nrof_epochs):
                train_or_test_epoch(sess, total_loss, regularization_losses, embeddings, labels, predictions, train_op, learning_rate_placeholder, 
                    epoch, emb_array_train, np.array(train_labels), args.batch_size, 0.01, True, True)
                if epoch % validate_every_n_epochs == 0 or epoch==args.max_nrof_epochs-1:
                    accuracy = train_or_test_epoch(sess, total_loss, regularization_losses, embeddings, labels, predictions, None, learning_rate_placeholder, 
                        epoch, emb_array_test, np.array(test_labels), args.batch_size, 0.0, False, True)
                    print('Test accuracy: %.3f' % accuracy)
                            

def train_or_test_epoch(sess, total_loss, regularization_losses, embeddings, labels, predictions, train_op, learning_rate_placeholder, 
        epoch, emb_array, labels_array, batch_size, learning_rate, is_training, print_results):
    nrof_images = len(labels_array)
    nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
    phase_string = ['Test ', 'Train']
    correct_predictions = np.zeros([nrof_images])
    for i in range(nrof_batches_per_epoch):
        start_time = time.time()
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        embeddings_batch = emb_array[start_index:end_index,:]
        labels_batch = labels_array[start_index:end_index]
        feed_dict = {embeddings: embeddings_batch, labels: labels_batch, learning_rate_placeholder: learning_rate}
        if is_training:
            loss, reg_loss, pred, _ = sess.run([total_loss, regularization_losses, predictions, train_op], feed_dict=feed_dict)
        else:
            loss, reg_loss, pred = sess.run([total_loss, regularization_losses, predictions], feed_dict=feed_dict)
        correct_predictions[start_index:end_index] = np.equal(np.argmax(pred, axis=1), np.array(labels_batch))
        duration = time.time() - start_time
        if print_results:
            print('%5s Epoch: [%d][%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\tAccuracy %.3f' %
                  (phase_string[is_training], epoch, i, duration, loss, np.sum(reg_loss), np.mean(correct_predictions[:end_index])))
    return np.mean(correct_predictions)
  
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    # Remove classes with less than min_nrof_images_per_class
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--use_split_dataset', 
        help='', action='store_true')
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=1000)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=1000)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
