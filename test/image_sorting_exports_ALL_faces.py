from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

import numpy as np
from scipy import misc as scp_misc
import tensorflow as tf

import facenet
import align.detect_face as detect_face
# from PIL import Image


def initialize_mtcnn(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def align_image(input_image, output_image, pnet, rnet, onet, image_size=182, margin=44, random_order=True,
                gpu_memory_fraction=1.0, debug=False, just_count=False):
    minsize = 20  # minimum size of face
    threshold = [0.7, 0.7, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    if not os.path.exists(output_image):
        try:
            img = scp_misc.imread(input_image)
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

            if just_count == True:
                return True, nrof_faces

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    det = np.squeeze(det)
                    counter = 0
                    scaled_list = []
                    for d in det:
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(d[0] - margin / 2, 0)
                        bb[1] = np.maximum(d[1] - margin / 2, 0)
                        bb[2] = np.minimum(d[2] + margin / 2, img_size[1])
                        bb[3] = np.minimum(d[3] + margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        scaled = scp_misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                        filename = "{}_{}.jpg".format(output_image.split(".")[0] + "image", str(counter))
                        scp_misc.imsave(filename, scaled)
                        scaled_list.append(scaled)
                        counter = counter +1
                    return True, scaled_list
                if nrof_faces == 1:
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = scp_misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    scp_misc.imsave(output_image, scaled)
                    return True, scaled
            else:
                if debug:
                    print('Unable to align "%s"' % input_image)

                return False, 1

def main(args):

    # TODO Check why this was previously being initialised inside the image loop
    file_to_facecount = dict()
    pnet, rnet, onet = initialize_mtcnn(0.8)
    for filename in os.listdir(args.input_dir):
        input_image = filename
        output_image = filename

        if os.path.isfile(os.path.join(args.input_dir, input_image)) == False:
            continue

        input_image = os.path.join(args.input_dir, input_image)
        output_image = os.path.join(args.output_dir,  output_image)

        _, result = align_image(input_image, output_image, pnet, rnet, onet, image_size=args.image_size, margin=args.margin, random_order=args.random_order,
                      gpu_memory_fraction=args.gpu_memory_fraction, debug=False, just_count=args.just_count)

        if args.just_count == True:
            file_to_facecount[filename] = result

    if args.just_count:
        json.dump(file_to_facecount, open(os.path.join(args.output_dir, args.count_file), "w"))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.',
                        default=1.0)
    parser.add_argument('--has_classes', dest='has_classes', action='store_true',
                        help='Input folder is split into class subfolders, and these should be replicated',
                        default=True)
    parser.add_argument('--no_classes', dest='has_classes', action='store_false',
                        help='Input folder is split into class subfolders, and these should be replicated',
                        default=True)
    parser.add_argument('--just_count', dest='just_count', action='store_true',
                        help='Just save out a JSON mapping filenames to counts of faces found',
                        default=False)
    parser.add_argument('--count_file', type=str,
                        help='Where to save counts of faces',
                        default="face_counts.json")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
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