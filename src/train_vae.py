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

"""Train a Variational Autoencoder
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
import time
import argparse
import facenet
import numpy as np
import h5py
import os
from datetime import datetime

def main(args):
  

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    log_file_name = os.path.join(model_dir, 'logs.h5')
        
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        train_set = facenet.get_dataset(args.data_dir)
        image_list, _ = facenet.get_image_paths_and_labels(train_set)
        
        images_tensor = ops.convert_to_tensor(image_list, dtype=tf.string)
        
        # Makes an input queue
        input_queue = tf.train.string_input_producer(images_tensor, shuffle=True)
    
        nrof_preprocess_threads = 4
        imagesx = []
        for _ in range(nrof_preprocess_threads):
            file_contents = tf.read_file(input_queue.dequeue())
            image = tf.image.decode_png(file_contents, channels=3)
            #pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            image = tf.image.per_image_standardization(image)
            imagesx.append([image])
    
        image_batch = tf.train.batch_join(
            imagesx, batch_size=args.batch_size,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=False)

        # Create encoder network
        mean, log_variance = encoder(image_batch, batch_norm_params, args.embedding_size)
        
        epsilon = tf.random_normal((args.batch_size, args.embedding_size))
        std = tf.exp(log_variance/2)
        latent_var = mean + epsilon * std
        
        # Create decoder network
        reconstructed = decoder(latent_var, batch_norm_params)
        
        # Create reconstruction loss (perceptual loss)
        #   This requires an instance of the facenet model
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(image_batch - reconstructed,2)))
        
        # Create KL divergence loss
        kl_loss = kl_divergence_loss(mean, log_variance)
        kl_loss_mean = tf.reduce_mean(kl_loss)
        
        total_loss = args.alfa*kl_loss_mean + args.beta*reconstruction_loss
        
        learning_rate = tf.train.exponential_decay(args.initial_learning_rate, global_step,
            args.learning_rate_decay_steps, args.learning_rate_decay_factor, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss)
        
        # Apply gradients
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Start running operations on the Graph
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            #feed_dict = {images_placeholder: np.random.normal(size=(batch_size,64,64,3))}
            
            log = {
                'total_loss': np.zeros((0,), np.float),
                'reconstruction_loss': np.zeros((0,), np.float),
                'kl_loss': np.zeros((0,), np.float),
                'learning_rate': np.zeros((0,), np.float),
                }
            
            for _ in range(args.max_nrof_steps):
                start_time = time.time()
                step, _, reconstruction_loss_, kl_loss_mean_, total_loss_, _, _, learning_rate_ = sess.run(
                      [global_step, train_op, reconstruction_loss, kl_loss_mean, total_loss, mean, log_variance, learning_rate])
                log['total_loss'] = np.append(log['total_loss'], total_loss_)
                log['reconstruction_loss'] = np.append(log['reconstruction_loss'], reconstruction_loss_)
                log['kl_loss'] = np.append(log['kl_loss'], kl_loss_mean_)
                log['learning_rate'] = np.append(log['learning_rate'], learning_rate_)

                duration = time.time() - start_time
                print('Step: %d \tTime: %.3f \trec_loss: %.3f \tkl_loss: %.3f \ttotal_loss: %.3f' % (step, duration, reconstruction_loss_, kl_loss_mean_, total_loss_))

                if step % args.save_every_n_steps==0:
                    print('Saving checkpoint file')
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
                    print('Saving log')
                    with h5py.File(log_file_name, 'w') as f:
                        for key, value in log.iteritems():
                            f.create_dataset(key, data=value)

                

def kl_divergence_loss(mean, log_variance):
    kl = 0.5 * tf.reduce_sum( tf.exp(log_variance) + tf.square(mean) - 1.0 - log_variance, reduction_indices = 1)
    return kl

def encoder(images, batch_norm_params, latent_variable_dim):
    # Note: change relu to leaky relu
    weight_decay = 0.0
    with tf.variable_scope('encoder'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            net = slim.conv2d(images, 32, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_1')
            net = slim.conv2d(net, 64, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_2')
            net = slim.conv2d(net, 128, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_3')
            net = slim.conv2d(net, 256, [4, 4], 2, activation_fn=tf.nn.relu, scope='Conv2d_4')
            net = slim.flatten(net)
            fc1 = slim.fully_connected(net, latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_1')
            fc2 = slim.fully_connected(net, latent_variable_dim, activation_fn=None, normalizer_fn=None, scope='Fc_2')
    return fc1, fc2
  
def decoder(latent, batch_norm_params):
    # Note: change relu to leaky relu
    weight_decay = 0.0
    with tf.variable_scope('decoder'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            net = slim.fully_connected(latent, 4096, activation_fn=None, normalizer_fn=None, scope='Fc_1')
            net = tf.reshape(net, [-1,4,4,256], name='Reshape')
            
            net = tf.image.resize_nearest_neighbor(net, size=(8,8), name='Upsample_1')
            net = slim.conv2d(net, 128, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_1')
    
            net = tf.image.resize_nearest_neighbor(net, size=(16,16), name='Upsample_2')
            net = slim.conv2d(net, 64, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_2')
    
            net = tf.image.resize_nearest_neighbor(net, size=(32,32), name='Upsample_3')
            net = slim.conv2d(net, 32, [3, 3], 1, activation_fn=tf.nn.relu, scope='Conv2d_3')
    
            net = tf.image.resize_nearest_neighbor(net, size=(64,64), name='Upsample_4')
            net = slim.conv2d(net, 3, [3, 3], 1, activation_fn=None, scope='Conv2d_4')
        
    return net

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/vae')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/home/david/datasets/casia/casia_maxpy_mtcnnpy_64')
    parser.add_argument('--max_nrof_steps', type=int,
        help='Number of steps to run.', default=50000)
    parser.add_argument('--save_every_n_steps', type=int,
        help='Number of steps between storing of model checkpoint and log files', default=1000)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=64)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=100)
    parser.add_argument('--initial_learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0005)
    parser.add_argument('--learning_rate_decay_steps', type=int,
        help='Number of steps between learning rate decay.', default=1)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--alfa', type=float,
        help='Kullback-Leibler divergence loss factor.', default=1.0)
    parser.add_argument('--beta', type=float,
        help='Reconstruction loss factor.', default=0.5)

    return parser.parse_args(argv)
  
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
