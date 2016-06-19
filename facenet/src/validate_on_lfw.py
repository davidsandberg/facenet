"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted
"""
import tensorflow as tf
import numpy as np
import importlib
import facenet
import os
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '~/models/facenet/20160514-234418',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string('lfw_pairs', '~/repo/facenet/data/lfw/pairs.txt',
                           """The file containing the pairs to use for validation.""")
tf.app.flags.DEFINE_string('file_ext', '.png',
                           """The file extension for the LFW dataset, typically .png or .jpg.""")
tf.app.flags.DEFINE_string('lfw_dir', '~/datasets/lfw/lfw_realigned/',
                           """Path to the data directory containing aligned face patches.""")
tf.app.flags.DEFINE_string('model_def', 'models.nn4',
                           """Model definition. Points to a module containing the definition of the inference graph.""")
tf.app.flags.DEFINE_integer('batch_size', 60,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_string('pool_type', 'MAX',
                          """The type of pooling to use for some of the inception layers {'MAX', 'L2'}.""")
tf.app.flags.DEFINE_boolean('use_lrn', False,
                          """Enables Local Response Normalization after the first layers of the inception network.""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """Random seed.""")

network = importlib.import_module(FLAGS.model_def, 'inference')

def main():
    
    pairs = read_pairs(os.path.expanduser(FLAGS.lfw_pairs))
    paths, actual_issame = get_paths(os.path.expanduser(FLAGS.lfw_dir), pairs)
    
    with tf.Graph().as_default():

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
          
        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
          
        # Build the inference graph
        embeddings = network.inference(images_placeholder, FLAGS.pool_type, FLAGS.use_lrn, 
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
    
            nrof_images = len(paths)
            nrof_batches = int(nrof_images / FLAGS.batch_size)  # Run forward pass on the remainder in the last batch
            emb_list = []
            for i in range(nrof_batches):
                start_time = time.time()
                paths_batch = paths[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                images = facenet.load_data(paths_batch, False, False, FLAGS.image_size)
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_list += sess.run([embeddings], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Calculated embeddings for batch %d of %d: time=%.3f seconds' % (i+1,nrof_batches, duration))
            emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
            
            thresholds = np.arange(0, 4, 0.01)
            embeddings1 = emb_array[0::2]
            embeddings2 = emb_array[1::2]
            tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), FLAGS.seed)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            facenet.plot_roc(fpr, tpr, 'NN4')
            

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+FLAGS.file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+FLAGS.file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+FLAGS.file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+FLAGS.file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

if __name__ == '__main__':
    main()
