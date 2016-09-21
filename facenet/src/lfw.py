from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import math

def validate(sess, actual_issame, seed, batch_size, embeddings, labels, nrof_folds=10):

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    nrof_images = len(actual_issame)*2
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_list = []
    lbl_list = []
    for _ in range(nrof_batches):
        emb, lbl = sess.run([embeddings, labels])
        emb_list.append(emb)
        lbl_list.append(lbl)
    emb_array = np.vstack(emb_list)
    lbl_array = np.hstack(lbl_list)
    emb_sorted = emb_array[lbl_array,:]

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_sorted[0::2]
    embeddings2 = emb_sorted[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), seed, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, seed, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
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
    return np.array(pairs)



