# MIT License
# 
# Copyright (c) 2017 David Sandberg
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

"""Base class for variational autoencoders containing an encoder and a decoder
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Vae(object):
  
    def __init__(self, latent_variable_dim, image_size):
        self.latent_variable_dim = latent_variable_dim
        self.image_size = image_size
        self.batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
  
    def encoder(self, images, is_training):
        # Must be overridden in implementation classes
        raise NotImplementedError
      
    def decoder(self, latent_var, is_training):
        # Must be overridden in implementation classes
        raise NotImplementedError

    def get_image_size(self):
        return self.image_size
        