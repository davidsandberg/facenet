"""Functions for building the face recognition network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from scipy import interpolate

def conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType, name, phase_train=True, use_batch_norm=True, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        kernel = tf.get_variable("weights", [kH, kW, nIn, nOut],
            initializer=tf.truncated_normal_initializer(stddev=1e-1),
            regularizer=l2_regularizer)
        cnv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
        
        if use_batch_norm:
            conv_bn = batch_norm(cnv, nOut, phase_train, 'batch_norm')
        else:
            conv_bn = cnv
        biases = tf.get_variable("biases", [nOut], initializer=tf.constant_initializer())
        bias = tf.nn.bias_add(conv_bn, biases)
        conv1 = tf.nn.relu(bias)
    return conv1

def affine(inpOp, nIn, nOut, name, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        weights = tf.get_variable("weights", [nIn, nOut],
            initializer=tf.truncated_normal_initializer(stddev=1e-1),
            regularizer=l2_regularizer)
        biases = tf.get_variable("biases", [nOut], initializer=tf.constant_initializer())
        affine1 = tf.nn.relu_layer(inpOp, weights, biases)
    return affine1

def l2_loss(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.
    Args:
      tensor: tensor to regularize.
      weight: an optional weight to modulate the loss.
      scope: Optional scope for op_scope.
    Returns:
      the L2 loss op.
    """
    with tf.op_scope([tensor], scope, 'l2_loss'):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
    return loss

def lppool(inpOp, pnorm, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        if pnorm == 2:
            pwr = tf.square(inpOp)
        else:
            pwr = tf.pow(inpOp, pnorm)
          
        subsamp = tf.nn.avg_pool(pwr,
                              ksize=[1, kH, kW, 1],
                              strides=[1, dH, dW, 1],
                              padding=padding)
        subsamp_sum = tf.mul(subsamp, kH*kW)
        
        if pnorm == 2:
            out = tf.sqrt(subsamp_sum)
        else:
            out = tf.pow(subsamp_sum, 1/pnorm)
    
    return out

def mpool(inpOp, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(inpOp,
                       ksize=[1, kH, kW, 1],
                       strides=[1, dH, dW, 1],
                       padding=padding)  
    return maxpool

def apool(inpOp, kH, kW, dH, dW, padding, name):
    with tf.variable_scope(name):
        avgpool = tf.nn.avg_pool(inpOp,
                              ksize=[1, kH, kW, 1],
                              strides=[1, dH, dW, 1],
                              padding=padding)
    return avgpool

def batch_norm(x, n_out, phase_train, name, affn=True):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    with tf.variable_scope(name):
  
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name=name+'/beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name=name+'/gamma', trainable=affn)
      
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                            beta, gamma, 1e-3, affn, name=name)
    return normed

def inception(inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, 
              phase_train=True, use_batch_norm=True, weight_decay=0.0):
  
    print('name = ', name)
    print('inputSize = ', inSize)
    print('kernelSize = {3,5}')
    print('kernelStride = {%d,%d}' % (ks,ks))
    print('outputSize = {%d,%d}' % (o2s2,o3s2))
    print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
    print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
    if (o4s2>0):
        o4 = o4s2
    else:
        o4 = inSize
    print('outputSize = ', o1s+o2s2+o3s2+o4)
    print()
    
    net = []
    
    with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
            if o1s>0:
                conv1 = conv(inp, inSize, o1s, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv1)
      
        with tf.variable_scope('branch2_3x3'):
            if o2s1>0:
                conv3a = conv(inp, inSize, o2s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv3 = conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 'SAME', 'conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv3)
      
        with tf.variable_scope('branch3_5x5'):
            if o3s1>0:
                conv5a = conv(inp, inSize, o3s1, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv5 = conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 'SAME', 'conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv5)
      
        with tf.variable_scope('branch4_pool'):
            if poolType=='MAX':
                pool = mpool(inp, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            elif poolType=='L2':
                pool = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)
            
            if o4s2>0:
                pool_conv = conv(pool, inSize, o4s2, 1, 1, 1, 1, 'SAME', 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
            else:
                pool_conv = pool
            net.append(pool_conv)
      
        incept = array_ops.concat(3, net, name=name)
    return incept

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)  # Summing over distances in each batch
        neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.sub(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def train(total_loss, global_step, optimizer, initial_learning_rate, 
    learning_rate_decay_steps, learning_rate_decay_factor, moving_average_decay):
    """Setup training for the FaceNet model.
  
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
  
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
        learning_rate_decay_steps, learning_rate_decay_factor, staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        else:
            raise ValueError('Invalid optimization algorithm')
    
        grads = opt.compute_gradients(total_loss)
      
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
  
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
  
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
  
    return train_op, grads

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.max(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = image.shape[1]/2
        sz2 = image_size/2
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        img_list[i] = img
    images = np.stack(img_list)
    return images

def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size<=nrof_examples:
        batch = image_data[j:j+batch_size,:,:,:]
    else:
        x1 = image_data[j:nrof_examples,:,:,:]
        x2 = image_data[0:nrof_examples-j,:,:,:]
        batch = np.vstack([x1,x2])
    batch_float = batch.astype(np.float32)
    return batch_float

def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch

def select_triplets(embeddings, num_per_class, image_data, people_per_batch, alpha):

    def dist(emb1, emb2):
        x = np.square(np.subtract(emb1, emb2))
        return np.sum(x, 0)
  
    nrof_images = image_data.shape[0]
    nrof_triplets = nrof_images - people_per_batch
    shp = [nrof_triplets, image_data.shape[1], image_data.shape[2], image_data.shape[3]]
    as_arr = np.zeros(shp)
    ps_arr = np.zeros(shp)
    ns_arr = np.zeros(shp)
    
    trip_idx = 0
    shuffle = np.arange(nrof_triplets)
    np.random.shuffle(shuffle)
    emb_start_idx = 0
    nrof_random_negs = 0
    for i in xrange(people_per_batch):
        n = num_per_class[i]
        for j in range(1,n):
            a_idx = emb_start_idx
            p_idx = emb_start_idx + j
            as_arr[shuffle[trip_idx]] = image_data[a_idx]
            ps_arr[shuffle[trip_idx]] = image_data[p_idx]
      
            # Select a semi-hard negative that has a distance
            #  further away from the positive exemplar.
            pos_dist = dist(embeddings[a_idx][:], embeddings[p_idx][:])
            sel_neg_idx = emb_start_idx
            while sel_neg_idx>=emb_start_idx and sel_neg_idx<=emb_start_idx+n-1:
                sel_neg_idx = (np.random.randint(1, 2**32) % nrof_images) -1  # Seems to give the same result as the lua implementation
                #sel_neg_idx = np.random.random_integers(0, nrof_images-1)
            sel_neg_dist = dist(embeddings[a_idx][:], embeddings[sel_neg_idx][:])
      
            random_neg = True
            for k in range(nrof_images):
                if k<emb_start_idx or k>emb_start_idx+n-1:
                    neg_dist = dist(embeddings[a_idx][:], embeddings[k][:])
                    if pos_dist<neg_dist and neg_dist<sel_neg_dist and np.abs(pos_dist-neg_dist)<alpha:
                        random_neg = False
                        sel_neg_dist = neg_dist
                        sel_neg_idx = k
            
            if random_neg:
                nrof_random_negs += 1
              
            ns_arr[shuffle[trip_idx]] = image_data[sel_neg_idx]
            #print('Triplet %d: (%d, %d, %d), pos_dist=%2.3f, neg_dist=%2.3f, sel_neg_dist=%2.3f' % (trip_idx, a_idx, p_idx, sel_neg_idx, pos_dist, neg_dist, sel_neg_dist))
            trip_idx += 1
          
        emb_start_idx += n
    
    triplets = (as_arr, ps_arr, ns_arr)
    
    return triplets, nrof_random_negs, nrof_triplets

  

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
  
    return image_paths, num_per_class

def load_model(model_file):
    tf.train.import_meta_graph(os.path.expanduser(model_file+'.meta'))
    ema = tf.train.ExponentialMovingAverage(1.0)
    restore_vars = {}
    for key, value in ema.variables_to_restore().items():
        if 'ExponentialMovingAverage' in key:
            restore_vars[key] = value
    saver = tf.train.Saver(restore_vars, name='ema_restore')
    saver.restore(tf.get_default_session(), os.path.expanduser(model_file))

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, seed, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, random_state=seed)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(folds):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def plot_roc(fpr, tpr, label):
    plt.plot(fpr, tpr, label=label)
    plt.title('Receiver Operating Characteristics')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], 'g--')
    plt.grid(True)
    plt.show()
  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, seed, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, random_state=seed)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(folds):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def store_revision_info(src_path, output_dir, arg_string):
  
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()
  
    # Get local changes
    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_diff = stdout.strip()
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)
