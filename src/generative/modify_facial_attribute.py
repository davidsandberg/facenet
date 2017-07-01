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

"""Calculate average latent variables (here called attribute vectors) 
for the different attributes in CelebA
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import argparse
import importlib
import facenet
import os
from datetime import datetime
import numpy as np
import math
import time
import h5py
from scipy import misc

def main(args):
  
    img_mean = np.array([134.10714722, 102.52040863, 87.15436554])
    img_stddev = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
    
    pretrained_model = '/home/david/vae/20170627-224709/model.ckpt-50000'
    
    fields, attribs_dict = read_annotations('/media/deep/datasets/CelebA/Anno/list_attr_celeba.txt')
    
    vae_def = importlib.import_module(args.vae_def)
    vae = vae_def.Vae(args.latent_var_size)
    
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
        
        # Get attributes for images
        nrof_attributes = len(fields)
        attribs_list = []
        for img in image_list:
            key = os.path.split(img)[1].split('.')[0]
            attr = attribs_dict[key]
            assert len(attr)==nrof_attributes
            attribs_list.append(attr)
            
        # Create the input queue
        input_queue = tf.train.slice_input_producer([image_list, attribs_list], num_epochs=1, shuffle=False)        
        
        nrof_preprocess_threads = 4
        imagesx = []
        for _ in range(nrof_preprocess_threads):
            filename = input_queue[0]
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)
            image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
            image = tf.image.resize_images(image, (64,64))
            image.set_shape((args.image_size, args.image_size, 3))
            attribs = input_queue[1]
            attribs.set_shape((nrof_attributes,))
            image = tf.cast(image, tf.float32)
            imagesx.append([image, attribs])
    
        image_batch, attribs_batch = tf.train.batch_join(
            imagesx, batch_size=args.batch_size, 
            shapes=[(args.image_size, args.image_size, 3), (nrof_attributes,)], enqueue_many=False,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)


        # Normalize
        image_batch_norm = (image_batch-img_mean) / img_stddev

        # Create encoder
        mean, log_variance = vae.encoder(image_batch_norm)
        
        epsilon = tf.random_normal((tf.shape(mean)[0], args.latent_var_size))
        std = tf.exp(log_variance/2)
        latent_var = mean + epsilon * std
        
        # Create decoder
        reconstructed_norm = vae.decoder(latent_var)
        
        # Un-normalize
        reconstructed = (reconstructed_norm*img_stddev) + img_mean

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
          
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
           
            nrof_images = len(image_list)
            nrof_batches = int(math.ceil(len(image_list) / args.batch_size))
            latent_vars = np.zeros((nrof_images, args.latent_var_size))
            attributes = np.zeros((nrof_images, nrof_attributes))
            for i in range(nrof_batches):
                start_time = time.time()
                latent_var_, attribs_ = sess.run([latent_var, attribs_batch])
                latent_vars[i:i+latent_var_.shape[0],:] = latent_var_
                attributes[i:i+attribs_.shape[0],:] = attribs_
                duration = time.time() - start_time
                print('Batch %d/%d: %.3f seconds' % (i+1, nrof_batches, duration))
            # NOTE: This will print the 'Out of range' warning if the last batch is not full,
            #  as described by https://github.com/tensorflow/tensorflow/issues/8330
             
            # Calculate average change in the latent variable when each attribute changes
            attribute_vectors = np.zeros((nrof_attributes, args.latent_var_size), np.float32)
            for i in range(nrof_attributes):
                pos_idx = np.argwhere(attributes[:,i]==1)[:,0]
                neg_idx = np.argwhere(attributes[:,i]==-1)[:,0]
                pos_avg = np.mean(latent_vars[pos_idx,:], 0)
                neg_avg = np.mean(latent_vars[neg_idx,:], 0)
                attribute_vectors[i,:] = pos_avg - neg_avg
            
            filename = os.path.join(model_dir, 'attribute_vectors.h5')
            print('Writing attribute vectors, latent variables and attributes to %s' % filename)
            mdict = {'latent_vars':latent_vars, 'attributes':attributes, 
                     'fields':fields, 'attribute_vectors':attribute_vectors }
            with h5py.File(filename, 'w') as f:
                for key, value in mdict.iteritems():
                    f.create_dataset(key, data=value)
                    
            with h5py.File(filename,'r') as f:
                latent_vars = np.array(f.get('latent_vars'))
                attributes = np.array(f.get('attributes'))
                fields = np.array(f.get('fields'))
                attribute_vectors = np.array(f.get('attribute_vectors'))

            # Reconstruct faces while adding varying amount of the selected attribute vector
            attribute_index = 31 # 31: 'Smiling'
            image_index = 0
            idx = np.argwhere(attributes[:,i]==-1)[image_index,0]
            nrof_interp_steps = 10
            sweep_latent_var = np.zeros((nrof_interp_steps, args.latent_var_size), np.float32)
            for i in range(nrof_interp_steps):
                sweep_latent_var[i,:] = latent_vars[idx,:] + 5.0*i/nrof_interp_steps*attribute_vectors[attribute_index,:]
                
            recon = sess.run(reconstructed, feed_dict={latent_var:sweep_latent_var})
            
            img = facenet.put_images_on_grid(recon, shape=(nrof_interp_steps,1))
            misc.imsave(os.path.join(model_dir, 'reconstructed_%06d.png' % idx), img)

                    
def read_annotations(filename):
    attribs = {}    
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i==0:
                continue  # First line is the number of entries in the file
            elif i==1:
                fields = line.strip().split() # Second line is the field names
            else:
                line = line.split()
                img_name = line[0].split('.')[0]
                img_attribs = map(int, line[1:])
                attribs[img_name] = img_attribs
    return fields, attribs

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/vae')
    parser.add_argument('--vae_def', type=str,
        help='Model definition for the variational autoencoder. Points to a module containing the definition.', 
        default='src.generative.models.dfc_vae')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/home/david/datasets/casia/casia_maxpy_mtcnnpy_64')
    parser.add_argument('--output_filename', type=str,
        help='Filename to use for the file containing the attribute vectors.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=64)
    parser.add_argument('--latent_var_size', type=int,
        help='Dimensionality of the latent variable.', default=100)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

    return parser.parse_args(argv)
  
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
