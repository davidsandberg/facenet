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

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models.network as network

def inference(images, keep_probability, phase_train=True, weight_decay=0.0):
    """ Define an inference network for face recognition based 
           on inception modules using batch normalization
    
    Args:
      images: The images to run inference on, dimensions batch_size x height x width x channels
      phase_train: True if batch normalization should operate in training mode
    """
    endpoints = {}
    net = network.conv(images, 3, 64, 7, 7, 2, 2, 'SAME', 'conv1_7x7', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv1'] = net
    net = network.mpool(net,  3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net
    net = network.conv(net,  64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net
    net = network.conv(net,  64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net
    net = network.mpool(net,  3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net
  
    net = network.inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3a'] = net
    net = network.inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'MAX', 'incept3b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3b'] = net
    net = network.inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3c'] = net
    
    net = network.inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'MAX', 'incept4a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4a'] = net
    net = network.inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4e'] = net
    
    net = network.inception(net, 1024, 1, 256, 96,  384, 0, 0, 3, 96,  1, 'MAX', 'incept5a', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5a'] = net
    net = network.inception(net, 736, 1, 256, 96, 384, 0, 0, 3, 96, 1, 'MAX', 'incept5b', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5b'] = net
    net = network.apool(net,  3, 3, 1, 1, 'VALID', 'pool6')
    endpoints['pool6'] = net
    net = tf.reshape(net, [-1, 736])
    endpoints['prelogits'] = net
    net = tf.nn.dropout(net, keep_probability)
    endpoints['dropout'] = net
    
    return net, endpoints
  