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
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def get_image_paths(facedir):
     image_paths = []
     if os.path.isdir(facedir):
         images = os.listdir(facedir)
         image_paths = [os.path.join(facedir,img) for img in images]
     return image_paths

def main(input_path, output_path, batch_size, model, image_size):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            #for filename in os.listdir(input_path):
             #   x = filename.split('_')[0]
                #directory = (output_path + "/" + x)

            # Read the file containing the pairs used for testing
            # pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            # paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = image_size
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')
            #batch_size = batch_size
            nrof_images = len(os.listdir(input_path))
            nrof_batches = 1 # int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i * batch_size
                print(start_index)
                end_index = min((i + 1) * batch_size, nrof_images)
                print(end_index)
                #paths_batch = paths[start_index:end_index]
                images = facenet.load_data(image_paths, False, False, image_size)
                print("I got this far!")
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                # ValueError: Cannot feed value of shape (3, 182, 182, 3) for Tensor u'input:0', which has shape '(?, 160, 160, 3)'
                #do I need to specify the shapt of the imput tensor somewhere?
                np.savetxt('/home/iolie/tensorflow/THORN/crop/TEST', emb_array, delimiter=",")

                ### emb array is all the image vectors in the order they were run,
                ### but how to ensure they went in in the same order? since list.os takes files randomly??
                ### additional column in emb array??
                ###



input_path = ('/home/iolie/tensorflow/THORN/Minisample') #+ "/" + "0A0A937C-016C-49E6-A9CA-480292B491BC")
output_path = ('/home/iolie/tensorflow/THORN/Minisample') #+ "/" + "0A0A937C-016C-49E6-A9CA-480292B491BC")

batch_size = len(os.listdir(input_path))
model = "/home/iolie/tensorflow/THORN/msft-cfs-face-recognition/analysis/analysis/facenet/temp_models/20170512-110547"
image_size = 160

image_paths = get_image_paths(input_path)

main(input_path, output_path, batch_size, model, image_size)
## this should output a TEST.txt file to the crop folder, with the few test images all contained within
## how big should my batch be??
