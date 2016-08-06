# boilerplate code
import numpy as np
from functools import partial
import PIL.Image

import tensorflow as tf
import matplotlib.pyplot as plt
import urllib2
import os
import zipfile

def main():
    # download pre-trained model by running the command below in a shell
    #  wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data/'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib2.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
  
    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
  
    model_fn = 'tensorflow_inception_graph.pb'
    
    # creating TensorFlow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
  
  
    # Helper functions for TF Graph visualization
    #pylint: disable=unused-variable
    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def
      
    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def
      
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()
        
    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5
    
    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
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
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
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
                g = sess.run(t_grad, {t_input:sub})
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
  
    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139 # picking some feature channel to visualize
    render_naive(T(layer)[:,:,:,channel])
    
    render_multiscale(T(layer)[:,:,:,channel])
   
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
     
    render_lapnorm(T(layer)[:,:,:,channel])
     
    render_lapnorm(T(layer)[:,:,:,65])
     
    render_lapnorm(T('mixed3b_1x1_pre_relu')[:,:,:,101])
     
    render_lapnorm(T(layer)[:,:,:,65]+T(layer)[:,:,:,139], octave_n=4)
     
     
    img0 = PIL.Image.open('pilatus800.jpg')
    img0 = np.float32(img0)
    showarray(img0/255.0)
     
    render_deepdream(tf.square(T('mixed4c')), img0)
     
    render_deepdream(T(layer)[:,:,:,139], img0)
    
  
if __name__ == '__main__':
    main()
