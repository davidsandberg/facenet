import os
import argparse
from sys import exit

import json

import tensorflow.compat.v1 as tf
from facenet.src.facenet import load_model, load_data


def get_image_paths(inpath):
    paths = []

    for (root, dirs, files) in os.walk(inpath):
        for f in (f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
            paths.append(os.path.join(root, f))

    return (paths)


def faces_to_vectors(inpath, modelpath, outpath, imgsize, batchsize=100):
    '''
    Given a folder and a model, loads images and performs forward pass to get a vector for each face
    results go to a JSON, with filenames mapped to their facevectors
    :param inpath: Where are your images? Must be cropped to faces (use MTCNN!)
    :param modelpath: Where is the tensorflow model we'll use to create the embedding?
    :param outpath: Full path to output file (better give it a JSON extension)
    :return: Number of faces converted to vectors
    '''
    results = dict()

    tf.disable_v2_behavior()
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:

            load_model(modelpath)
            mdl = None

            image_paths = get_image_paths(inpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Let's do them in batches, don't want to run out of memory
            for i in range(0, len(image_paths), batchsize):
                images = load_data(image_paths=image_paths[i:i+batchsize], do_random_crop=False, do_random_flip=False, image_size=imgsize, do_prewhiten=True)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}

                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                for j in range(0, len(emb_array)):
                    relpath = os.path.relpath(image_paths[i+j], inpath)
                    results[relpath] = emb_array[j].tolist()

    # All done, save for later!
    json.dump(results, open(outpath, "w"))

    return len(results.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", help="Folder with images - png/jpg/jpeg - of faces", type=str, required=True)
    parser.add_argument("--outpath", help="Full path to where you want the results JSON", type=str, required=True)
    parser.add_argument("--mdlpath", help="Where to find the Tensorflow model to use for the embedding", type=str, required=True)
    parser.add_argument("--imgsize", help="Size of images to use", type=int, default=160, required=False)
    args = parser.parse_args()

    num_images_processed = faces_to_vectors(args.inpath, args.mdlpath, args.outpath, args.imgsize)
    if num_images_processed > 0:
        print("Converted " + str(num_images_processed) + " images to face vectors.")
    else:
        print("No images were processed - are you sure that was the right path? [" + args.inpath + "]")
        return False

    return True


if __name__ == main():
    if main() is False:
        exit(-1)

    exit(0)
