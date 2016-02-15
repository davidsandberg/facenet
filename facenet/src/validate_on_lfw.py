"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted
"""
import math
import tensorflow as tf
import numpy as np
import facenet
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.platform import gfile

import os
import sys
import time

tf.app.flags.DEFINE_string('model_dir', '/home/david/logs/openface/20160214-213259',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string('lfw_pairs', '/home/david/repo/facenet/data/lfw/pairs.txt',
                           """The file containing the pairs to use for validation.""")
tf.app.flags.DEFINE_string('lfw_dir', '/home/david/datasets/lfw_aligned/',
                           """Path to the data directory containing aligned face patches.""")
tf.app.flags.DEFINE_integer('batch_size', 90,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")

FLAGS = tf.app.flags.FLAGS


def create_graph(graphdef_filename):
    """"Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with gfile.FastGFile(graphdef_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')



def main():
    
    # Creates graph from saved GraphDef
    #  NOTE: This does not work at the moment. Needs tensorflow to store variables in the graph_def.
    #create_graph(os.path.join(FLAGS.model_dir, 'graphdef', 'graph_def.pb'))
    
    pairs = read_pairs(FLAGS.lfw_pairs)
    paths, actual_issame = get_paths(FLAGS.lfw_dir, pairs)
    
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
    
        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='Input')
        
        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        # Build the inference graph
        embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=phase_train_placeholder)
        
        # Create a saver
        saver = tf.train.Saver(tf.all_variables())
    
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
        with tf.Session() as sess:
    
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
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
                images = facenet.load_data(paths_batch)
                #padded = np.lib.pad(images, (10,0,0,0), 'constant', constant_values=(0,0,0,0))
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_list += sess.run([embeddings], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Calculated embeddings for batch %d of %d: time=%.3f seconds' % (i+1,nrof_batches, duration))
            emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
            
            thresholds = np.arange(0, 4, 0.01)
            tpr, fpr = calculate_roc(thresholds, emb_array, actual_issame)
            
            plt.plot(fpr, tpr)
            plt.show()
            

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.png')
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.png')
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.png')
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.png')
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

def calculate_roc(thresholds, embeddings, actual_issame):
    tpr_array = []
    fpr_array = []
    predict_issame = [None] * len(actual_issame)
    for threshold in thresholds:
        tp = tn = fp = fn = 0
        x1 = embeddings[0::2]
        x2 = embeddings[1::2]
        diff = np.subtract(x1, x2)
        dist = np.sum(np.square(diff),1)
        
        for i in range(dist.size):
            predict_issame[i] = dist[i] < threshold
            if predict_issame[i] and actual_issame[i]:
                tp += 1
            elif predict_issame[i] and not actual_issame[i]:
                fp += 1
            elif not predict_issame[i] and not actual_issame[i]:
                tn += 1
            elif not predict_issame[i] and actual_issame[i]:
                fn += 1
    
        if tp + fn == 0:
            tpr = 0
        else:
            tpr = float(tp) / float(tp + fn)
        if fp + tn == 0:
            fpr = 0
        else:
            fpr = float(fp) / float(fp + tn)
            
        tpr_array.append(tpr)
        fpr_array.append(fpr)

    return tpr_array, fpr_array

if __name__ == '__main__':
    main()