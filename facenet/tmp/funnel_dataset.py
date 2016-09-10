"""Performs face alignment and stores face thumbnails in the output directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import facenet
import subprocess
from contextlib import contextmanager
import tempfile
import shutil
import numpy as np

@contextmanager
def TemporaryDirectory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def main(args):
    funnel_cmd = 'funnelReal'
    funnel_model = 'people.train'

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the output directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    np.random.shuffle(dataset)
    # Scale the image such that the face fills the frame when cropped to crop_size
    #scale = float(args.face_size) / args.image_size
    with TemporaryDirectory() as tmp_dir:
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            tmp_output_class_dir = os.path.join(tmp_dir, cls.name)
            if not os.path.exists(output_class_dir) and not os.path.exists(tmp_output_class_dir):
                print('Aligning class %s:' % cls.name)
                tmp_filenames = []
                if not os.path.exists(tmp_output_class_dir):
                    os.makedirs(tmp_output_class_dir)
                input_list_filename = os.path.join(tmp_dir, 'input_list.txt')
                output_list_filename = os.path.join(tmp_dir, 'output_list.txt')
                input_file = open(input_list_filename, 'w')
                output_file = open(output_list_filename,'w')
                for image_path in cls.image_paths:
                    filename = os.path.split(image_path)[1]
                    input_file.write(image_path+'\n')
                    output_filename = os.path.join(tmp_output_class_dir, filename)
                    output_file.write(output_filename+'\n')
                    tmp_filenames.append(output_filename)
                input_file.close()
                output_file.close()
                cmd = args.funnel_dir+funnel_cmd + ' ' + input_list_filename + ' ' + args.funnel_dir+funnel_model + ' ' + output_list_filename
                subprocess.call(cmd, shell=True)
                
                # Resize and crop images
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                scale = 1.0
                for tmp_filename in tmp_filenames:
                    img = misc.imread(tmp_filename)
                    img_scale = misc.imresize(img, scale)
                    sz1 = img.shape[1]/2
                    sz2 = args.image_size/2
                    img_crop = img_scale[int(sz1-sz2):int(sz1+sz2),int(sz1-sz2):int(sz1+sz2),:]
                    filename = os.path.splitext(os.path.split(tmp_filename)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename+'.png')
                    print('Saving image %s' % output_filename)
                    misc.imsave(output_filename, img_crop)
                    
                # Remove tmp directory with images
                shutil.rmtree(tmp_output_class_dir)
                
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('funnel_dir', type=str, help='Directory containing the funnelReal binary and the people.train model file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=110)
    parser.add_argument('--face_size', type=int,
        help='Size of the face thumbnail (height, width) in pixels.', default=96)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
