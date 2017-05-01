"""Test invariance to translation, scaling and rotation on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
This requires test images to be cropped a bit wider than the normal to give some room for the transformations.
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
import matplotlib.pyplot as plt
from scipy import misc
import os
import sys
import math

def main(args):
    
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
    result_dir = '../data/'
    plt.ioff()  # Disable interactive plotting mode
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
    
            # Load the model
            print('Loading model "%s"' % args.model_file)
            facenet.load_model(args.model_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            image_size = int(images_placeholder.get_shape()[1])
            
            # Run test on LFW to check accuracy for different horizontal/vertical translations of input images
            if args.nrof_offsets>0:
                step = 3
                offsets = np.asarray([x*step for x in range(-args.nrof_offsets//2+1, args.nrof_offsets//2+1)])
                horizontal_offset_accuracy = [None] * len(offsets)
                for idx, offset in enumerate(offsets):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, 
                        paths, actual_issame, translate_images, (offset,0), 60, args.orig_image_size, args.seed)
                    print('Hoffset: %1.3f  Accuracy: %1.3f+-%1.3f' % (offset, np.mean(accuracy), np.std(accuracy)))
                    horizontal_offset_accuracy[idx] = np.mean(accuracy)
                vertical_offset_accuracy = [None] * len(offsets)
                for idx, offset in enumerate(offsets):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, 
                        paths, actual_issame, translate_images, (0,offset), 60, args.orig_image_size, args.seed)
                    print('Voffset: %1.3f  Accuracy: %1.3f+-%1.3f' % (offset, np.mean(accuracy), np.std(accuracy)))
                    vertical_offset_accuracy[idx] = np.mean(accuracy)
                fig = plt.figure(1)
                plt.plot(offsets, horizontal_offset_accuracy, label='Horizontal')
                plt.plot(offsets, vertical_offset_accuracy, label='Vertical')
                plt.legend()
                plt.grid(True)
                plt.title('Translation invariance on LFW')
                plt.xlabel('Offset [pixels]')
                plt.ylabel('Accuracy')
#                plt.show()
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_translation.png'))
                save_result(offsets, horizontal_offset_accuracy, os.path.join(result_dir, 'invariance_translation_horizontal.txt'))
                save_result(offsets, vertical_offset_accuracy, os.path.join(result_dir, 'invariance_translation_vertical.txt'))

            # Run test on LFW to check accuracy for different rotation of input images
            if args.nrof_angles>0:
                step = 3
                angles = np.asarray([x*step for x in range(-args.nrof_offsets//2+1, args.nrof_offsets//2+1)])
                rotation_accuracy = [None] * len(angles)
                for idx, angle in enumerate(angles):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, 
                        paths, actual_issame, rotate_images, angle, 60, args.orig_image_size, args.seed)
                    print('Angle: %1.3f  Accuracy: %1.3f+-%1.3f' % (angle, np.mean(accuracy), np.std(accuracy)))
                    rotation_accuracy[idx] = np.mean(accuracy)
                fig = plt.figure(2)
                plt.plot(angles, rotation_accuracy)
                plt.grid(True)
                plt.title('Rotation invariance on LFW')
                plt.xlabel('Angle [deg]')
                plt.ylabel('Accuracy')
#                plt.show()
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_rotation.png'))
                save_result(angles, rotation_accuracy, os.path.join(result_dir, 'invariance_rotation.txt'))

            # Run test on LFW to check accuracy for different scaling of input images
            if args.nrof_scales>0:
                step = 0.05
                scales = np.asarray([x*step+1 for x in range(-args.nrof_offsets//2+1, args.nrof_offsets//2+1)])
                scale_accuracy = [None] * len(scales)
                for scale_idx, scale in enumerate(scales):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, 
                        paths, actual_issame, scale_images, scale, 60, args.orig_image_size, args.seed)
                    print('Scale: %1.3f  Accuracy: %1.3f+-%1.3f' % (scale, np.mean(accuracy), np.std(accuracy)))
                    scale_accuracy[scale_idx] = np.mean(accuracy)
                fig = plt.figure(3)
                plt.plot(scales, scale_accuracy)
                plt.grid(True)
                plt.title('Scale invariance on LFW')
                plt.xlabel('Scale')
                plt.ylabel('Accuracy')
#                plt.show()
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_scale.png'))
                save_result(scales, scale_accuracy, os.path.join(result_dir, 'invariance_scale.txt'))
                
def save_result(aug, acc, filename):
    with open(filename, "w") as f:
        for i in range(aug.size):
            f.write('%6.4f %6.4f\n' % (aug[i], acc[i]))
            
def evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, 
        paths, actual_issame, augument_images, aug_value, batch_size, orig_image_size, seed):
    nrof_images = len(paths)
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_list = []
    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, orig_image_size)
        images_aug = augument_images(images, aug_value, image_size)
        feed_dict = { images_placeholder: images_aug, phase_train_placeholder: False }
        emb_list += sess.run([embeddings], feed_dict=feed_dict)
    emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
    
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]
    _, _, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), seed)
    return accuracy

def scale_images(images, scale, image_size):
    images_scale_list = [None] * images.shape[0]
    for i in range(images.shape[0]):
        images_scale_list[i] = misc.imresize(images[i,:,:,:], scale)
    images_scale = np.stack(images_scale_list,axis=0)
    sz1 = images_scale.shape[1]/2
    sz2 = image_size/2
    images_crop = images_scale[:,(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
    return images_crop

def rotate_images(images, angle, image_size):
    images_list = [None] * images.shape[0]
    for i in range(images.shape[0]):
        images_list[i] = misc.imrotate(images[i,:,:,:], angle)
    images_rot = np.stack(images_list,axis=0)
    sz1 = images_rot.shape[1]/2
    sz2 = image_size/2
    images_crop = images_rot[:,(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
    return images_crop

def translate_images(images, offset, image_size):
    h, v = offset
    sz1 = images.shape[1]/2
    sz2 = image_size/2
    images_crop = images[:,(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return images_crop

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_file', type=str, 
        help='File containing the model parameters as well as the model metagraph (with extension ".meta")', 
        default='~/models/facenet/20160514-234418/model.ckpt-500000')
    parser.add_argument('--nrof_offsets', type=int,
        help='Number of horizontal and vertical offsets to evaluate.', default=21)
    parser.add_argument('--nrof_angles', type=int,
        help='Number of angles to evaluate.', default=21)
    parser.add_argument('--nrof_scales', type=int,
        help='Number of scales to evaluate.', default=21)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='~/datasets/lfw/lfw_realigned/')
    parser.add_argument('--orig_image_size', type=int,
        help='Image size (height, width) in pixels of the original (uncropped/unscaled) images.', default=224)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
