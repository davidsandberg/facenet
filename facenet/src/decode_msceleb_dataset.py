"""Decode the MsCelebV1 dataset in TSV (tab separated values) format downloaded from
https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import numpy as np
import base64
import sys
import os
import cv2
import argparse
import facenet


# File format: text files, each line is an image record containing 6 columns, delimited by TAB.
# Column1: Freebase MID
# Column2: Query/Name
# Column3: ImageSearchRank
# Column4: ImageURL
# Column5: PageURL
# Column6: ImageData_Base64Encoded

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
  
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
  
    # Store some git revision info in a text file in the output directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    
    i = 0
    for f in args.tsv_files:
        for line in f:
            fields = line.split('\t')
            class_dir = fields[0]
            img_name = fields[1] + '-' + fields[4] + '.' + args.output_format
            img_string = fields[6]
            img_dec_string = base64.b64decode(img_string)
            img_data = np.fromstring(img_dec_string, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.cv.CV_LOAD_IMAGE_COLOR) #pylint: disable=maybe-no-member
            if args.size>0:
                img = misc.imresize(img, (args.size, args.size), interp='bilinear')
            full_class_dir = os.path.join(output_dir, class_dir)
            if not os.path.exists(full_class_dir):
                os.mkdir(full_class_dir)
            full_path = os.path.join(full_class_dir, img_name)
            cv2.imwrite(full_path, img) #pylint: disable=maybe-no-member
            print('%8d: %s' % (i, full_path))
            i += 1

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str, help='Output base directory for the image dataset')
    parser.add_argument('tsv_files', type=argparse.FileType('r'), nargs='+', help='Input TSV file name(s)')
    parser.add_argument('--size', type=int, help='Images are resized to the given size', default=-1)
    parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])

    main(parser.parse_args())

