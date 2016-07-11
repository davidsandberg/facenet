"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import os
import align_dlib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_file', '~/models/facenet/20160514-234418/model.ckpt-500000',
                           """File containing the model parameters as well as the model metagraph (with extension '.meta')""")
tf.app.flags.DEFINE_string('dlib_face_predictor', '../data/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_string('image1', '', """First image to compare.""")
tf.app.flags.DEFINE_string('image2', '', """Second image to compare.""")

def main():
    align = align_dlib.AlignDlib(os.path.expanduser(FLAGS.dlib_face_predictor))
    image_paths = [FLAGS.image1, FLAGS.image2]
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            print('Loading model "%s"' % FLAGS.model_file)
            facenet.load_model(FLAGS.model_file)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            image_size = int(images_placeholder.get_shape()[1])

            # Run forward pass to calculate embeddings
            images = load_and_align_data(image_paths, image_size, align, landmarkIndices)
            feed_dict = { images_placeholder: images, phase_train_placeholder: False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            dist = np.sqrt(np.mean(np.square(np.subtract(emb[0,:], emb[1,:]))))
            print('Distance between the embeddings: %3.6f' % dist)
            
def load_and_align_data(image_paths, image_size, align, landmarkIndices):
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(image_paths[i])
        aligned = align.align(image_size, img, landmarkIndices=landmarkIndices, skipMulti=True)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

if __name__ == '__main__':
    main()
