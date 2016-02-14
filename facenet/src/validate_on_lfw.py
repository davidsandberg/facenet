#!/usr/bin/env python3
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implements the standard LFW verification experiment.

import math
import tensorflow as tf
import numpy as np
import facenet
#import pandas as pd
from scipy.interpolate import interp1d

#from sklearn.cross_validation import KFold
#from sklearn.metrics import accuracy_score

import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.style.use('bmh')

from tensorflow.python.platform import gfile

import os
import sys
import time

#from scipy import arange

tf.app.flags.DEFINE_string('model_dir', '/home/david/logs/openface/20160214-134915',
                           """Directory the graph definitation and checkpoint is stored.""")
tf.app.flags.DEFINE_string('lfw_pairs', '/home/david/repo/facenet/data/lfw/pairs.txt',
                           """Directory the graph definitation and checkpoint is stored.""")
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
            
            xxx = 1

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


def evalThresholdAccuracy(embeddings, pairs, threshold):
    y_true = []
    y_predict = []
    for pair in pairs:
        (x1, x2, actual_same) = getEmbeddings(pair, embeddings)
        diff = x1 - x2
        dist = np.dot(diff.T, diff)
        predict_same = dist < threshold
        y_predict.append(predict_same)
        y_true.append(actual_same)

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = accuracy_score(y_true, y_predict)
    return accuracy


def findBestThreshold(thresholds, embeddings, pairsTrain):
    bestThresh = bestThreshAcc = 0
    for threshold in thresholds:
        accuracy = evalThresholdAccuracy(embeddings, pairsTrain, threshold)
        if accuracy >= bestThreshAcc:
            bestThreshAcc = accuracy
            bestThresh = threshold
        else:
            # No further improvements.
            return bestThresh
    return bestThresh


def verifyExp(workDir, pairs, embeddings):
    print("  + Computing accuracy.")
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = arange(0, 4, 0.01)

    if os.path.exists("{}/accuracies.txt".format(workDir)):
        print("{}/accuracies.txt already exists. Skipping processing.".format(workDir))
    else:
        accuracies = []
        with open("{}/accuracies.txt".format(workDir), "w") as f:
            f.write('fold, threshold, accuracy\n')
            for idx, (train, test) in enumerate(folds):
                fname = "{}/l2-roc.fold-{}.csv".format(workDir, idx)
                writeROC(fname, thresholds, embeddings, pairs[test])

                bestThresh = findBestThreshold(
                    thresholds, embeddings, pairs[train])
                accuracy = evalThresholdAccuracy(
                    embeddings, pairs[test], bestThresh)
                accuracies.append(accuracy)
                f.write('{}, {:0.2f}, {:0.2f}\n'.format(
                    idx, bestThresh, accuracy))
            avg = np.mean(accuracies)
            std = np.std(accuracies)
            f.write('\navg, {:0.4f} +/- {:0.4f}\n'.format(avg, std))
            print('    + {:0.4f}'.format(avg))


def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)


def plotOpenFaceROC(workDir, plotFolds=True, color=None):
    fs = []
    for i in range(10):
        rocData = pd.read_csv("{}/l2-roc.fold-{}.csv".format(workDir, i))
        fs.append(interp1d(rocData['fpr'], rocData['tpr']))
        x = np.linspace(0, 1, 1000)
        if plotFolds:
            foldPlot, = plt.plot(x, fs[-1](x), color='grey', alpha=0.5)
        else:
            foldPlot = None

    fprs = []
    tprs = []
    for fpr in np.linspace(0, 1, 1000):
        tpr = 0.0
        for f in fs:
            v = f(fpr)
            if math.isnan(v):
                v = 0.0
            tpr += v
        tpr /= 10.0
        fprs.append(fpr)
        tprs.append(tpr)
    if color:
        meanPlot, = plt.plot(fprs, tprs, color=color)
    else:
        meanPlot, = plt.plot(fprs, tprs)
    AUC = getAUC(fprs, tprs)
    return foldPlot, meanPlot, AUC


def plotVerifyExp(workDir, tag):
    print("Plotting.")

    fig, ax = plt.subplots(1, 1)

    openbrData = pd.read_csv("comparisons/openbr.v1.1.0.DET.csv")
    openbrData['Y'] = 1 - openbrData['Y']
    # brPlot = openbrData.plot(x='X', y='Y', legend=True, ax=ax)
    brPlot, = plt.plot(openbrData['X'], openbrData['Y'])
    brAUC = getAUC(openbrData['X'], openbrData['Y'])

    foldPlot, meanPlot, AUC = plotOpenFaceROC(workDir, color='k')

    humanData = pd.read_table(
        "comparisons/kumar_human_crop.txt", header=None, sep=' ')
    humanPlot, = plt.plot(humanData[1], humanData[0])
    humanAUC = getAUC(humanData[1], humanData[0])

    deepfaceData = pd.read_table(
        "comparisons/deepface_ensemble.txt", header=None, sep=' ')
    dfPlot, = plt.plot(deepfaceData[1], deepfaceData[0], '--',
                       alpha=0.75)
    deepfaceAUC = getAUC(deepfaceData[1], deepfaceData[0])

    # baiduData = pd.read_table(
    #     "comparisons/BaiduIDLFinal.TPFP", header=None, sep=' ')
    # bPlot, = plt.plot(baiduData[1], baiduData[0])
    # baiduAUC = getAUC(baiduData[1], baiduData[0])

    eigData = pd.read_table(
        "comparisons/eigenfaces-original-roc.txt", header=None, sep=' ')
    eigPlot, = plt.plot(eigData[1], eigData[0])
    eigAUC = getAUC(eigData[1], eigData[0])

    ax.legend([humanPlot, dfPlot, brPlot, eigPlot,
               meanPlot, foldPlot],
              ['Human, Cropped [AUC={:.3f}]'.format(humanAUC),
               # 'Baidu [{:.3f}]'.format(baiduAUC),
               'DeepFace Ensemble [{:.3f}]'.format(deepfaceAUC),
               'OpenBR v1.1.0 [{:.3f}]'.format(brAUC),
               'Eigenfaces [{:.3f}]'.format(eigAUC),
               'OpenFace {} [{:.3f}]'.format(tag, AUC),
               'OpenFace {} folds'.format(tag)],
              loc='lower right')

    plt.plot([0, 1], color='k', linestyle=':')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    # fig.savefig(os.path.join(workDir, "roc.pdf"))
    fig.savefig(os.path.join(workDir, "roc.png"))

if __name__ == '__main__':
    main()