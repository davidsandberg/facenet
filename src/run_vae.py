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

"""Generate images or latent variables using a Variational Autoencoder
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import argparse
import facenet
import os
from datetime import datetime
import matplotlib.pyplot as plt

def main(args):
  
    #pretrained_model = '/home/david/vae/20170617-232027/model.ckpt-50000'
    pretrained_model = '/home/david/vae/20170620-224017/model.ckpt-41000'
    
    # Create encoder
    # Create decoder
    # Load parameters 
  
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
    
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, model_dir, ' '.join(sys.argv))
        
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        
        train_set = facenet.get_dataset(args.data_dir)
        image_list, _ = facenet.get_image_paths_and_labels(train_set)
        
        images_placeholder = tf.placeholder(tf.float32, shape=(None,64,64,3), name='input')
        
        # Create encoder network
        mean, log_variance = encoder(images_placeholder, batch_norm_params, args.embedding_size)
        
        epsilon = tf.random_normal((args.batch_size, args.embedding_size))
        std = tf.exp(log_variance/2)
        latent_var = mean + epsilon * std
        
        # Create decoder network
        reconstructed = decoder(latent_var, batch_norm_params)
        
        

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        
        #facenet_saver = tf.train.Saver(get_facenet_variables_to_restore())

        # Start running operations on the Graph
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
          
            image_paths = image_list[0:128]
            imgs = facenet.load_data(image_paths, False, False, 64, True)
            recon_ = sess.run([reconstructed], feed_dict={images_placeholder:imgs})
            
            xxx = 1
            
            plt.imshow((recon_[0][0,:,:,:]+127)*50)
            plt.imshow((imgs[0,:,:,:]+127)*10)
            



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
    parser.add_argument('--reconstruction_loss_type', type=str, choices=['PLAIN', 'PERCEPTUAL'],
        help='The type of reconstruction loss to use', default='PERCEPTUAL')
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
        help='Initial learning rate.', default=0.0005)
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
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')

    return parser.parse_args(argv)
  
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
