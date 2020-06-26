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
from facenet import load_data,load_img,load_model,to_rgb
#import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import cv2

from align import detect_face
from datetime import datetime
from scipy import ndimage
from scipy.misc import imsave 
from scipy.spatial.distance import cosine
import pickle
#face_cascade = cv2.CascadeClassifier('out/face/haarcascade_frontalface_default.xml')
parser = argparse.ArgumentParser()
    

parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

args = parser.parse_args()

# def align_face(img,pnet, rnet, onet):

# 		        minsize = 20 # minimum size of face
# 		        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
# 		        factor = 0.709 # scale factor

#                 print("before img.size == 0")	
#                         if img.size == 0:
#                                 print("empty array")
# 				return False,img,[0,0,0,0]

#                         if img.ndim<2:
#                             print('Unable to align')

#                         if img.ndim == 2:
#                             img = to_rgb(img)

#                         img = img[:,:,0:3]
	
# 			bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

#                         nrof_faces = bounding_boxes.shape[0]

                        
#                         if nrof_faces==0:
#                             return False,img,[0,0,0,0]
#                         else:
#                             det = bounding_boxes[:,0:4]
#                             det_arr = []
#                             img_size = np.asarray(img.shape)[0:2]
#                             if nrof_faces>1:
#                                 if args.detect_multiple_faces:
#                                     for i in range(nrof_faces):
#                                         det_arr.append(np.squeeze(det[i]))
#                                 else:
#                                     bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
#                                     img_center = img_size / 2
#                                     offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
#                                     offset_dist_squared = np.sum(np.power(offsets,2.0),0)
#                                     index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
#                                     det_arr.append(det[index,:])
#                             else:
#                                 det_arr.append(np.squeeze(det))
#                             if len(det_arr)>0:
#                                     faces = []
#                                     bboxes = []
# 		                    for i, det in enumerate(det_arr):
# 		                        det = np.squeeze(det)
# 		                        bb = np.zeros(4, dtype=np.int32)
# 		                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
# 		                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
# 		                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
# 		                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
# 		                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
# 		                        scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
# 		                        misc.imsave("cropped.png", scaled)
#                                         faces.append(scaled)
#                                         bboxes.append(bb)
# 		                        print("leaving align face")
# 		                    return True,faces,bboxes
			

# def identify_person(image_vector, feature_array, k=9):
# 	top_k_ind = np.argsort([np.linalg.norm(image_vector-pred_row) for ith_row, pred_row in enumerate(feature_array.values())])[:k]
#     result = feature_array.keys()[top_k_ind[0]]
#     acc = np.linalg.norm(image_vector-feature_array.values()[top_k_ind[0]])
#     return result, acc
            


def recognize_face(sess,pnet, rnet, onet,feature_array):
    # Get input and output tensors
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    image_size = args.image_size
    embedding_size = embeddings.get_shape()[1]

    cap = cv2.VideoCapture(-1)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        if(gray.size > 0):
            print(gray.size)
            response, faces,bboxs = align_face(gray,pnet, rnet, onet)
            print(response)
            if (response == True):
                    for i, image in enumerate(faces):
                            bb = bboxs[i]
                            images = load_img(image, False, False, image_size)
                            feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                            result, accuracy = identify_person(feature_vector, feature_array,8)
                            print(result.split("/")[2])
                            print(accuracy)

                            if accuracy < 0.9:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"Hello "+result.split("/")[2],(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            else:
                                cv2.rectangle(gray,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
                                W = int(bb[2]-bb[0])//2
                                H = int(bb[3]-bb[1])//2
                                cv2.putText(gray,"WHO ARE YOU ?",(bb[0]+W-(W//2),bb[1]-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                            del feature_vector

            cv2.imshow('img',gray)
        else:
            continue
                

def align_face(img,pnet, rnet, onet):
	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709 # scale factor
	if img.size == 0:
		print("empty array")
		return False,img,[0,0,0,0]

	if img.ndim<2:
		print('Unable to align')

	if img.ndim == 2:
		img = to_rgb(img)

	img = img[:,:,0:3]

	bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
	nrof_faces = bounding_boxes.shape[0]

	if nrof_faces==0:
		return False,img,[0,0,0,0]
	else:
		det = bounding_boxes[:,0:4]
        det_arr = list()
        img_size = np.asarray(img.shape)[0:2]
        
        if nrof_faces>1:
        	if args.detect_multiple_faces:
        		for i in range(nrof_faces):
        			det_arr.append(np.squeeze(det[i]))

        	else:
        		bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])

        else:
        	det_arr.append(np.squeeze(det))

        if len(det_arr)>0:
        	faces = []
            bboxes = []

        for i, det in enumerate(det_arr):
        	det = np.squeeze(det)
		    bb = np.zeros(4, dtype=np.int32)
		    bb[0] = np.maximum(det[0]-args.margin/2, 0)
		    bb[1] = np.maximum(det[1]-args.margin/2, 0)
		    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
		    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
		    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
		    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
		    misc.imsave("cropped.png", scaled)
		    faces.append(scaled)
            bboxes.append(bb)
            print("leaving align face")

        return True,faces,bboxes


                           