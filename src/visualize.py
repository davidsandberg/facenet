"""Visualize individual feature channels and their combinations to explore the space of patterns learned by the neural network
Based on http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sys
import argparse
import tensorflow as tf
import importlib
from scipy import misc

def main(args):
  
    network = importlib.import_module(args.model_def, 'inference')

    # Start with a gray image with a little noise
    np.random.seed(seed=args.seed)
    img_noise = np.random.uniform(size=(args.image_size,args.image_size,3)) + 100.0
  
    sess = tf.Session()
  
    t_input = tf.placeholder(np.float32, shape=(args.image_size,args.image_size,3), name='input') # define the input tensor
    image_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-image_mean, 0)
     
    # Build the inference graph
    network.inference(t_preprocessed, 1.0, 
            phase_train=True, weight_decay=0.0)
      
    # Create a saver for restoring variables
    saver = tf.train.Saver(tf.global_variables())
  
    # Restore the parameters
    saver.restore(sess, args.model_file)
  
    layers = [op.name for op in tf.get_default_graph().get_operations() if op.type=='Conv2D']
    feature_nums = {layer: int(T(layer).get_shape()[-1]) for layer in layers}
  
    print('Number of layers: %d' % len(layers))
  
    for layer in sorted(feature_nums.keys()):
        print('%s%d' % ((layer+': ').ljust(40), feature_nums[layer]))
  
    # Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
    # to have non-zero gradients for features with negative initial activations.
    layer = 'InceptionResnetV1/Repeat_2/block8_3/Conv2d_1x1/Conv2D'
    #layer = 'incept4b/in4_conv1x1_31/Conv2D'
    result_dir = '../data/'
    print('Number of features in layer "%s": %d' % (layer, feature_nums[layer]))
    channels = range(feature_nums[layer])
    np.random.shuffle(channels)
    for i in range(32):
        print('Rendering feature %d' % channels[i])
        channel = channels[i]
        img = render_naive(sess, t_input, T(layer)[:,:,:,channel], img_noise)
        filename = '%s_%03d.png' % (layer.replace('/', '_'), channel)
        misc.imsave(os.path.join(result_dir, filename), img)
  

def T(layer):
    '''Helper for getting layer output tensor'''
    return tf.get_default_graph().get_tensor_by_name('%s:0' % layer)

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def render_naive(sess, t_input, t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    for _ in range(iter_n):
        g, _ = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
    return visstd(img)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_file', type=str, 
        help='Directory containing the graph definition and checkpoint files.')
    parser.add_argument('--model_def', type=str, 
        help='Model definition. Points to a module containing the definition of the inference graph.',
        default='models.nn4')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
