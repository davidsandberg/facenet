"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
import tensorflow as tf
import numpy as np
import facenet
import lfw
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_file', '~/models/facenet/20160514-234418/model.ckpt-500000',
                           """File containing the model parameters as well as the model metagraph (with extension '.meta')""")
tf.app.flags.DEFINE_string('lfw_pairs', '../data/pairs.txt',
                           """The file containing the pairs to use for validation.""")
tf.app.flags.DEFINE_string('file_ext', '.png',
                           """The file extension for the LFW dataset, typically .png or .jpg.""")
tf.app.flags.DEFINE_string('lfw_dir', '~/datasets/lfw/lfw_realigned/',
                           """Path to the data directory containing aligned face patches.""")
tf.app.flags.DEFINE_integer('seed', 666, """Random seed.""")

def main(argv=None):
  
    with tf.Graph().as_default():

        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(FLAGS.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(FLAGS.lfw_dir), pairs, FLAGS.file_ext)
            
            # Load the model
            print('Loading model "%s"' % FLAGS.model_file)
            facenet.load_model(FLAGS.model_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            tpr, fpr, accuracy, val, val_std, far = lfw.validate(sess, 
                paths, actual_issame, FLAGS.seed, 60, 
                images_placeholder, phase_train_placeholder, embeddings)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            
            facenet.plot_roc(fpr, tpr, 'NN4')
            
if __name__ == '__main__':
    tf.app.run()
