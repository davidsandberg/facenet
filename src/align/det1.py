#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.align.network import Network
#import Network

class PNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='PReLU1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='PReLU2')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='PReLU3')
             .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
             .softmax(3,name='prob1'))

        (self.feed('PReLU3')
             .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))
        