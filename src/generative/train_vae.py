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
import sys
import time
import importlib
import argparse
import facenet
import numpy as np
import h5py
import os
from datetime import datetime
from scipy import misc
from six import iteritems

def main(args):
  
    img_mean = np.array([134.10714722, 102.52040863, 87.15436554])
    img_stddev = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
  
    vae_def = importlib.import_module(args.vae_def)
    vae = vae_def.Vae(args.latent_var_size)
    gen_image_size = vae.get_image_size()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    log_file_name = os.path.join(model_dir, 'logs.h5')
    
    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(model_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, model_dir, ' '.join(sys.argv))
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        train_set = facenet.get_dataset(args.data_dir)
        image_list, _ = facenet.get_image_paths_and_labels(train_set)
        
        # Create the input queue
        input_queue = tf.train.string_input_producer(image_list, shuffle=True)
    
        nrof_preprocess_threads = 4
        image_per_thread = []
        for _ in range(nrof_preprocess_threads):
            file_contents = tf.read_file(input_queue.dequeue())
            image = tf.image.decode_image(file_contents, channels=3)
            image = tf.image.resize_image_with_crop_or_pad(image, args.input_image_size, args.input_image_size)
            image.set_shape((args.input_image_size, args.input_image_size, 3))
            image = tf.cast(image, tf.float32)
            #pylint: disable=no-member
            image_per_thread.append([image])
    
        images = tf.train.batch_join(
            image_per_thread, batch_size=args.batch_size,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=False)
        
        # Normalize
        images_norm = (images-img_mean) / img_stddev

        # Resize to appropriate size for the encoder 
        images_norm_resize = tf.image.resize_images(images_norm, (gen_image_size,gen_image_size))
        
        # Create encoder network
        mean, log_variance = vae.encoder(images_norm_resize, True)
        
        epsilon = tf.random_normal((tf.shape(mean)[0], args.latent_var_size))
        std = tf.exp(log_variance/2)
        latent_var = mean + epsilon * std
        
        # Create decoder network
        reconstructed_norm = vae.decoder(latent_var, True)
        
        # Un-normalize
        reconstructed = (reconstructed_norm*img_stddev) + img_mean
        
        # Create reconstruction loss
        if args.reconstruction_loss_type=='PLAIN':
            images_resize = tf.image.resize_images(images, (gen_image_size,gen_image_size))
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(images_resize - reconstructed,2)))
        elif args.reconstruction_loss_type=='PERCEPTUAL':
            network = importlib.import_module(args.model_def)

            reconstructed_norm_resize = tf.image.resize_images(reconstructed_norm, (args.input_image_size,args.input_image_size))

            # Stack images from both the input batch and the reconstructed batch in a new tensor 
            shp = [-1] + images_norm.get_shape().as_list()[1:]
            input_images = tf.reshape(tf.stack([images_norm, reconstructed_norm_resize], axis=0), shp)
            _, end_points = network.inference(input_images, 1.0, 
                phase_train=False, bottleneck_layer_size=128, weight_decay=0.0)

            # Get a list of feature names to use for loss terms
            feature_names = args.loss_features.replace(' ', '').split(',')

            # Calculate L2 loss between original and reconstructed images in feature space
            reconstruction_loss_list = []
            for feature_name in feature_names:
                feature_flat = slim.flatten(end_points[feature_name])
                image_feature, reconstructed_feature = tf.unstack(tf.reshape(feature_flat, [2,args.batch_size,-1]), num=2, axis=0)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(image_feature-reconstructed_feature, 2)), name=feature_name+'_loss')
                reconstruction_loss_list.append(reconstruction_loss)
            # Sum up the losses in for the different features
            reconstruction_loss = tf.add_n(reconstruction_loss_list, 'reconstruction_loss')
        else:
            pass
        
        # Create KL divergence loss
        kl_loss = kl_divergence_loss(mean, log_variance)
        kl_loss_mean = tf.reduce_mean(kl_loss)
        
        total_loss = args.alfa*kl_loss_mean + args.beta*reconstruction_loss
        
        learning_rate = tf.train.exponential_decay(args.initial_learning_rate, global_step,
            args.learning_rate_decay_steps, args.learning_rate_decay_factor, staircase=True)
        
        # Calculate gradients and make sure not to include parameters for the perceptual loss model
        opt = tf.train.AdamOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss, var_list=get_variables_to_train())
        
        # Apply gradients
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        
        facenet_saver = tf.train.Saver(get_facenet_variables_to_restore())

        # Start running operations on the Graph
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            
            if args.reconstruction_loss_type=='PERCEPTUAL':
                if not args.pretrained_model:
                    raise ValueError('A pretrained model must be specified when using perceptual loss')
                pretrained_model_exp = os.path.expanduser(args.pretrained_model)
                print('Restoring pretrained model: %s' % pretrained_model_exp)
                facenet_saver.restore(sess, pretrained_model_exp)
          
            log = {
                'total_loss': np.zeros((0,), np.float),
                'reconstruction_loss': np.zeros((0,), np.float),
                'kl_loss': np.zeros((0,), np.float),
                'learning_rate': np.zeros((0,), np.float),
                }
            
            step = 0
            print('Running training')
            while step < args.max_nrof_steps:
                start_time = time.time()
                step += 1
                save_state = step>0 and (step % args.save_every_n_steps==0 or step==args.max_nrof_steps)
                if save_state:
                    _, reconstruction_loss_, kl_loss_mean_, total_loss_, learning_rate_, rec_ = sess.run(
                          [train_op, reconstruction_loss, kl_loss_mean, total_loss, learning_rate, reconstructed])
                    img = facenet.put_images_on_grid(rec_, shape=(16,8))
                    misc.imsave(os.path.join(model_dir, 'reconstructed_%06d.png' % step), img)
                else:
                    _, reconstruction_loss_, kl_loss_mean_, total_loss_, learning_rate_ = sess.run(
                          [train_op, reconstruction_loss, kl_loss_mean, total_loss, learning_rate])
                log['total_loss'] = np.append(log['total_loss'], total_loss_)
                log['reconstruction_loss'] = np.append(log['reconstruction_loss'], reconstruction_loss_)
                log['kl_loss'] = np.append(log['kl_loss'], kl_loss_mean_)
                log['learning_rate'] = np.append(log['learning_rate'], learning_rate_)

                duration = time.time() - start_time
                print('Step: %d \tTime: %.3f \trec_loss: %.3f \tkl_loss: %.3f \ttotal_loss: %.3f' % (step, duration, reconstruction_loss_, kl_loss_mean_, total_loss_))

                if save_state:
                    print('Saving checkpoint file')
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
                    print('Saving log')
                    with h5py.File(log_file_name, 'w') as f:
                        for key, value in iteritems(log):
                            f.create_dataset(key, data=value)

def get_variables_to_train():
    train_variables = []
    for var in tf.trainable_variables():
        if 'Inception' not in var.name:
            train_variables.append(var)
    return train_variables

def get_facenet_variables_to_restore():
    facenet_variables = []
    for var in tf.global_variables():
        if var.name.startswith('Inception'):
            if 'Adam' not in var.name:
                facenet_variables.append(var)
    return facenet_variables

def kl_divergence_loss(mean, log_variance):
    kl = 0.5 * tf.reduce_sum( tf.exp(log_variance) + tf.square(mean) - 1.0 - log_variance, reduction_indices = 1)
    return kl

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('vae_def', type=str,
        help='Model definition for the variational autoencoder. Points to a module containing the definition.')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.')
    parser.add_argument('pretrained_model', type=str,
        help='Pretrained model to use to calculate features for perceptual loss.')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/vae')
    parser.add_argument('--loss_features', type=str,
        help='Comma separated list of features to use for perceptual loss. Features should be defined ' +
          'in the end_points dictionary.', default='Conv2d_1a_3x3,Conv2d_2a_3x3, Conv2d_2b_3x3')
    parser.add_argument('--reconstruction_loss_type', type=str, choices=['PLAIN', 'PERCEPTUAL'],
        help='The type of reconstruction loss to use', default='PERCEPTUAL')
    parser.add_argument('--max_nrof_steps', type=int,
        help='Number of steps to run.', default=50000)
    parser.add_argument('--save_every_n_steps', type=int,
        help='Number of steps between storing of model checkpoint and log files', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--input_image_size', type=int,
        help='Image size of input images (height, width) in pixels. If perceptual loss is used this ' 
        + 'should be the input image size for the perceptual loss model', default=160)
    parser.add_argument('--latent_var_size', type=int,
        help='Dimensionality of the latent variable.', default=100)
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
    
    return parser.parse_args(argv)
  
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
