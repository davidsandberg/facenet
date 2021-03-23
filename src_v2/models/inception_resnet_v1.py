# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Block35(tf.keras.Model):
    def __init__(self):
        super(Block35, self).__init__(name='')        
        # Branch_0
        self.conv1 = tf.keras.layers.Conv2D(32, (1, 1),padding='same')        
        # Branch_1
        self.conv2a = tf.keras.layers.Conv2D(32, (1, 1),padding='same')
        self.conv2b = tf.keras.layers.Conv2D(32, (3, 3),padding='same')
        # Branch_2
        self.conv3a = tf.keras.layers.Conv2D(32, (1, 1),padding='same')
        self.conv3b = tf.keras.layers.Conv2D(32, (3, 3),padding='same')
        self.conv3c = tf.keras.layers.Conv2D(32, (3, 3),padding='same')
        # Up
        self.convup = tf.keras.layers.Conv2D(32, (1, 1),padding='same')
        

    def call(self, input_tensor, scale = 1.0, activation_fn=tf.nn.relu,):
        # Branch_0
        x = self.conv1(input_tensor)
        # Branch_1
        y_1 = self.conv2a(input_tensor)
        y_2 = self.conv2b(y_1)
        # Branch_2
        z_1 = self.conv3a(input_tensor)
        z_2 = self.conv3b(z_1)
        z_3 = self.conv3c(z_2)

        mixed = tf.concat([x, y_2, z_3], 3)
        up = tf.keras.layers.Conv2D(input_tensor.get_shape()[3], (1,1))(mixed)

        input_tensor += scale * up       
        if activation_fn:
            input_tensor = activation_fn(input_tensor)

        
        return input_tensor