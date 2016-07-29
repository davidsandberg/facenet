"""Visualize individual feature channels and their combinations to explore the space of patterns learned by the neural network
Based on http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""

import os
import numpy as np
from functools import partial
import PIL.Image
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import importlib

def main(args):
  
    network = importlib.import_module(args.model_def, 'inference')

    # Start with a gray image with a little noise
    np.random.seed(seed=args.seed)
    img_noise = np.random.uniform(size=(96,96,3)) + 100.0
  
    sess = tf.Session()
  
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    image_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-image_mean, 0)
     
    # Placeholder for phase_train
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
  
    # Build the inference graph
    _ = network.inference(t_preprocessed, 'MAX', False, 1.0, phase_train=phase_train_placeholder)
      
    # Create a saver for restoring variable averages
    ema = tf.train.ExponentialMovingAverage(1.0)
    restore_vars = ema.variables_to_restore()
    saver = tf.train.Saver(restore_vars)
  
    # Restore the parameters
    ckpt = tf.train.get_checkpoint_state(os.path.expanduser(args.model_dir))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('Checkpoint not found')
  
    # Helper functions for TF Graph visualization
    
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()
               
    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5
    
    def T(layer):
        '''Helper for getting layer output tensor'''
        return tf.get_default_graph().get_tensor_by_name('%s:0' % layer)
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img, phase_train_placeholder:False})
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
        showarray(visstd(img))
        
    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=sess)
            return wrapper
        return wrap
    
    # Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)
    
    
    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub, phase_train_placeholder:False})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)    
      
    def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        for octave in range(octave_n):
            if octave>0:
                hw = np.float32(img.shape[:2])*octave_scale
                img = resize(img, np.int32(hw))
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                # normalizing the gradient, so the same step size should work 
                g /= g.std()+1e-8         # for different layers and networks
                img += g*step
            showarray(visstd(img))
            
    def lap_split(img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
            hi = img-lo2
        return lo, hi
    
    def lap_split_n(img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for _ in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]
    
    def lap_merge(levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img
    
    def normalize_std(img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)
    
    def lap_normalize(img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img,0)
        tlevels = lap_split_n(img, scale_n)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels)
        return out[0,:,:,:]
  
    def render_lapnorm(t_obj, img0=img_noise, visfunc=visstd,
                       iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        # build the laplacian normalization graph
        lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))
    
        img = img0.copy()
        for octave in range(octave_n):
            if octave>0:
                hw = np.float32(img.shape[:2])*octave_scale
                img = resize(img, np.int32(hw))
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                g = lap_norm_func(g)
                img += g*step
            showarray(visfunc(img))
  
    def render_deepdream(t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
        # split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
            showarray(img/255.0)
  
    layers = [op.name for op in tf.get_default_graph().get_operations() if op.type=='Conv2D']
    feature_nums = {layer: int(T(layer).get_shape()[-1]) for layer in layers}
  
    print('Number of layers: %d' % len(layers))
  
    for layer in sorted(feature_nums.keys()):
        print('%s%d' % ((layer+': ').ljust(40), feature_nums[layer]))
  
    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    layer = 'incept5b/in4_conv1x1_55/Conv2D'
    print('Number of features in layer "%s": %d' % (layer, feature_nums[layer]))
  
#   channels = range(feature_nums[layer])
#   np.random.shuffle(channels)
#   for i in range(16):
#     print('Rendering feature %d' % channels[i])
#     plt.subplot(4,4,i+1)
#     render_multiscale(T(layer)[:,:,:,channels[i]])
  
    channel = 0 # picking some feature channel to visualize
    
    render_naive(T(layer)[:,:,:,channel])
  
    render_multiscale(T(layer)[:,:,:,channel])
  
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
    
    render_lapnorm(T(layer)[:,:,:,channel])
    
    render_lapnorm(T(layer)[:,:,:,65]+T(layer)[:,:,:,139], octave_n=4)
    
    # Not tested yet
    img0 = PIL.Image.open('pilatus800.jpg')
    img0 = np.float32(img0)
    showarray(img0/255.0)
    render_deepdream(tf.square(T('mixed4c')), img0)
    render_deepdream(T(layer)[:,:,:,139], img0)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the graph definition and checkpoint files.')
    parser.add_argument('--model_def', type=str, 
        help='Model definition. Points to a module containing the definition of the inference graph.',
        default='models.nn4')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
