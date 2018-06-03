from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import os
import sys
import math
import pickle
import zerorpc

from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange

import logging
logging.basicConfig()

'''
  Change these two constants to the model and classifier you are using / trained.
'''
MODEL = "/path/to/model.pb"
CLASSIFIER_FILENAME = "/path/to/your/classifier.pkl"
'''
  Point to an example image to classify - need one to setup the system once on
  startup! Change this!
'''
preload_image = "/path/to/some/image.jpg"



MARGIN = 44
GPU_MEMORY_FRACTION = 1.0
IMAGE_SIZE = 160


bbs = []
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


img_li = []
imgTmp = misc.imread(os.path.expanduser(preload_image))
img_sz = np.asarray(imgTmp.shape)[0:2]


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        facenet.load_model(MODEL)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


    bounding_boxes, _ = align.detect_face.detect_face(imgTmp, minsize, pnet, rnet, onet, threshold, factor)
        count_per_image = len(bounding_boxes)
        for j in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[j,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-MARGIN/2, 0)
            bb[1] = np.maximum(det[1]-MARGIN/2, 0)
            bb[2] = np.minimum(det[2]+MARGIN/2, img_sz[1])
            bb[3] = np.minimum(det[3]+MARGIN/2, img_sz[0])
            cropped = imgTmp[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_li.append(prewhitened)
            bbs.append(bounding_boxes[j])
        images = np.stack(img_li)

        # Run forward pass to calculate embeddings
        feed_dict = { images_placeholder: images , phase_train_placeholder:False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
	print("Initiation complete - you can now call from Node.js using RPC!")


class RPCCom(object):
    def classifyFile(self, file_to_process):
        # Print to console the file we are dealing with for debugging purposes.
        print(file_to_process)
        img_list = []
        img = misc.imread(os.path.expanduser(file_to_process))
        img_size = np.asarray(img.shape)[0:2]

        with tf.Graph().as_default():
            with sess.as_default():
                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                count_per_image = len(bounding_boxes)

                for j in range(len(bounding_boxes)):
                    det = np.squeeze(bounding_boxes[j,0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-MARGIN/2, 0)
                    bb[1] = np.maximum(det[1]-MARGIN/2, 0)
                    bb[2] = np.minimum(det[2]+MARGIN/2, img_size[1])
                    bb[3] = np.minimum(det[3]+MARGIN/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    aligned = misc.imresize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    img_list.append(prewhitened)
                    bbs.append(bounding_boxes[j])
                images = np.stack(img_list)

                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(CLASSIFIER_FILENAME)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                #print predictions
                prediction_str = ""
                # print("\npeople in image %s :" %(file_to_process))
                prediction_str = prediction_str + class_names[best_class_indices[0]] + "," + str(best_class_probabilities[0]) + ";"

                # Print Prediction and optional Bounding Boxes...
                print(prediction_str);
                # print(bbs)

                return "%s" % prediction_str


def main():
    # Setup RPC server.
    s = zerorpc.Server(RPCCom())
    s.bind("tcp://0.0.0.0:4242")
    s.run()

main()
