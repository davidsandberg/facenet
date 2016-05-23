"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

from scipy import misc
import tensorflow as tf
import numpy as np
import facenet
import os
import align_dlib

tf.app.flags.DEFINE_string('model_dir', '~/models/facenet/20160514-234418',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string('dlib_face_predictor', '~/repo/openface/models/dlib/shape_predictor_68_face_landmarks.dat',
                           """File containing the dlib face predictor.""")
tf.app.flags.DEFINE_string('image1', '', """First image to compare.""")
tf.app.flags.DEFINE_string('image2', '', """Second image to compare.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_string('pool_type', 'MAX',
                          """The type of pooling to use for some of the inception layers {'MAX', 'L2'}.""")
tf.app.flags.DEFINE_boolean('use_lrn', False,
                          """Enables Local Response Normalization after the first layers of the inception network.""")

FLAGS = tf.app.flags.FLAGS

def main():
    align = align_dlib.AlignDlib(os.path.expanduser(FLAGS.dlib_face_predictor))
    batch_size = 2
    image_paths = [FLAGS.image1, FLAGS.image2]
    landmarkIndices = align_dlib.AlignDlib.OUTER_EYES_AND_NOSE

    with tf.Graph().as_default():

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
          
        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
          
        # Build the inference graph
        embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, FLAGS.pool_type, FLAGS.use_lrn, 
                                                       1.0, phase_train=phase_train_placeholder)
          
        # Create a saver for restoring variable averages
        ema = tf.train.ExponentialMovingAverage(1.0)
        saver = tf.train.Saver(ema.variables_to_restore())
        
        with tf.Session() as sess:
      
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('Checkpoint not found')
    
            images = load_and_align_data(image_paths, FLAGS.image_size, align, landmarkIndices)
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
