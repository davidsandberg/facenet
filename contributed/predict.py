

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rishi Rai
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
#----------------------------------------------------


import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange

def main(args):
  
    images, cout_per_image, nrof_samples = load_and_align_data(args.image_files,args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

       with tf.Session() as sess:
      
            # Load the model
                facenet.load_model(args.model)
            # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                classifier_filename_exp = os.path.expanduser(args.classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                k=0     
	    #print predictions       
                for i in range(nrof_samples):
                    print("\npeople in image %s :" %(args.image_files[i]))
                    for j in range(cout_per_image[i]):
                        print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                        k+=1
                    
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [] 
    count_per_image = []
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        count_per_image.append(len(bounding_boxes))
        for j in range(len(bounding_boxes)):	
                det = np.squeeze(bounding_boxes[j,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)		
    images = np.stack(img_list)
    return images, count_per_image, nrof_samples

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_files', type=str, nargs='+', help='Path(s) of the image(s)')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

