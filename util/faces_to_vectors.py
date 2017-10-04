import os
import ntpath
import argparse
from sys import exit

import json

import numpy as np
from PIL import Image


def gen_image_paths(inpath):
    for file in os.listdir(inpath):
        if os.path.isfile(os.path.join(inpath, file)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) is False:
                continue

            yield os.path.join(inpath, file)

    return


def map_image_to_vector(mdl, image):
    return None


def faces_to_vectors(inpath, modelpath, outpath):
    '''
    Given a folder and a model, loads images and performs forward pass to get a vector for each face
    results go to a JSON, with filenames mapped to their facevectors
    :param inpath: Where are your images? Must be cropped to faces (use MTCNN!)
    :param modelpath: Where is the tensorflow model we'll use to create the embedding?
    :param outpath: Full path to output file (better give it a JSON extension)
    :return: Number of faces converted to vectors
    '''
    results = dict()

    # TODO load model
    mdl = None

    for image_path in gen_image_paths(inpath):
        # TODO load image
        img = Image.open(open(inpath))
        results[ntpath.basename(image_path)] = map_image_to_vector(mdl, img)

    # All done, save for later!
    json.dump(results, open(outpath, "w"))

    return len(results.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", help="Folder with images - png/jpg/jpeg - of faces", type=str, required=True)
    parser.add_argument("outpath", help="Full path to where you want the results JSON", type=str, required=True)
    parser.add_argument("mdlpath", help="Where to find the Tensorflow model to use for the embedding", type=str, required=True)

    args = parser.parse_args()
    num_images_processed = faces_to_vectors(args.inpath, args.mdlpath, args.outpath)
    if num_images_processed > 0:
        print("Converted " + str(num_images_processed) + " to face vectors.")
    else:
        print("No images were processed - are you sure that was the right path? [" + args.inpath + "]")
        return False

    return True


if __name__ == main():
    if main() is False:
        exit(-1)

    exit(0)
