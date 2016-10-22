"""Download the VGG face dataset from URLs given by http://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz
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

from scipy import misc
import numpy as np
from skimage import io
import sys
import argparse
import os
import socket
from urllib2 import HTTPError, URLError
from httplib import HTTPException

def main(args):
    socket.setdefaulttimeout(30)
    textfile_names = os.listdir(args.dataset_descriptor)
    for textfile_name in textfile_names:
        if textfile_name.endswith('.txt'):
            with open(os.path.join(args.dataset_descriptor, textfile_name), 'rt') as f:
                lines = f.readlines()
            dir_name = textfile_name.split('.')[0]
            class_path = os.path.join(args.dataset_descriptor, dir_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            for line in lines:
                x = line.split(' ')
                filename = x[0]
                url = x[1]
                box = np.rint(np.array(map(float, x[2:6])))  # x1,y1,x2,y2
                image_path = os.path.join(args.dataset_descriptor, dir_name, filename+'.'+args.output_format)
                error_path = os.path.join(args.dataset_descriptor, dir_name, filename+'.err')
                if not os.path.exists(image_path) and not os.path.exists(error_path):
                    try:
                        img = io.imread(url, mode='RGB')
                    except (HTTPException, HTTPError, URLError, IOError, ValueError, IndexError, OSError) as e:
                        error_message = '{}: {}'.format(url, e)
                        save_error_message_file(error_path, error_message)
                    else:
                        try:
                            if img.ndim == 2:
                                img = to_rgb(img)
                            if img.ndim != 3:
                                raise ValueError('Wrong number of image dimensions')
                            hist = np.histogram(img, 255, density=True)
                            if hist[0][0]>0.9 and hist[0][254]>0.9:
                                raise ValueError('Image is mainly black or white')
                            else:
                                # Crop image according to dataset descriptor
                                img_cropped = img[box[1]:box[3],box[0]:box[2],:]
                                # Scale to 256x256
                                img_resized = misc.imresize(img_cropped, (args.image_size,args.image_size))
                                # Save image as .png
                                misc.imsave(image_path, img_resized)
                        except ValueError as e:
                            error_message = '{}: {}'.format(url, e)
                            save_error_message_file(error_path, error_message)
            
def save_error_message_file(filename, error_message):
    print(error_message)
    with open(filename, "w") as textfile:
        textfile.write(error_message)
          
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_descriptor', type=str, 
        help='Directory containing the text files with the image URLs. Image files will also be placed in this directory.')
    parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=256)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
