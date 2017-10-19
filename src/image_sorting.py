from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet

import align.detect_face as detect_face
import tensorflow as tf
# from PIL import Image
from scipy import misc


def initialize_mtcnn(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def align_image(input_image, output_image, pnet, rnet, onet, image_size=182, margin=44, random_order=True,
                gpu_memory_fraction=1.0, debug=False):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    if not os.path.exists(output_image):
        try:
            img = misc.imread(input_image)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(input_image, e)
            if debug:
                print(errorMessage)
        else:
            if img.ndim < 2:
                if debug:
                    print('Unable to align "%s"' % image_path)
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det = det[index, :]
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                misc.imsave(output_image, scaled)
                return True, scaled
            else:
                if debug:
                    print('Unable to align "%s"' % image_path)

                return False, 1

input_path = '/home/iolie/tensorflow/THORN/bp_aug_2017_dataset'
output_path = '/home/iolie/tensorflow/THORN/CROPPED_NO_SORT'

for filename in os.listdir(input_path):
   # x = filename.split('_')[0]
   # directory = (output_path + "/" + x)
   # if not os.path.exists(directory):
    #    os.makedirs(directory)

    input_image = filename
    output_image = filename
    pnet, rnet, onet = initialize_mtcnn(0.8)

    input_image = os.path.join(input_path, input_image)
    output_image = os.path.join(output_path,  output_image)

    align_image(input_image, output_image, pnet, rnet, onet, image_size=160, margin=44, random_order=True,
                  gpu_memory_fraction=1.0, debug=False)


#
#print(ads)
#print("bleh")
#print(os.listdir(path))

#
# for filename in os.listdir(path):
#     print(filename)
#     x = filename.split('_')[0]
#     ads.append(x)
#     directory = (path + "/" + x)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     #shutil.copy(newpath + "/" + filename, directory)