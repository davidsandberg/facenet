"""Load the VGG Face model into TensorFlow.
Download the model from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
and point to the file 'vgg_face.mat'
"""
import numpy as np
from scipy import io
import tensorflow as tf

def load(filename, images):
    #filename = '../data/vgg_face_matconvnet/data/vgg_face.mat'
    vgg16 = io.loadmat(filename)
    vgg16Layers = vgg16['net'][0][0]['layers']
    
    # A function to get the weights of the VGG layers
    def vbbWeights(layerNumber):
        W = vgg16Layers[0][layerNumber][0][0][2][0][0]
        W = tf.constant(W)
        return W
     
    def vbbConstants(layerNumber):
        b = vgg16Layers[0][layerNumber][0][0][2][0][1].T
        b = tf.constant(np.reshape(b, (b.size)))
        return b
    
    modelGraph = {}
    modelGraph['input'] = images
    
    modelGraph['conv1_1'] = tf.nn.conv2d(modelGraph['input'], filter = vbbWeights(0), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu1_1'] = tf.nn.relu(modelGraph['conv1_1'] + vbbConstants(0))
    modelGraph['conv1_2'] = tf.nn.conv2d(modelGraph['relu1_1'], filter = vbbWeights(2), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu1_2'] = tf.nn.relu(modelGraph['conv1_2'] + vbbConstants(2))
    modelGraph['pool1'] = tf.nn.max_pool(modelGraph['relu1_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    modelGraph['conv2_1'] = tf.nn.conv2d(modelGraph['pool1'], filter = vbbWeights(5), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu2_1'] = tf.nn.relu(modelGraph['conv2_1'] + vbbConstants(5))
    modelGraph['conv2_2'] = tf.nn.conv2d(modelGraph['relu2_1'], filter = vbbWeights(7), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu2_2'] = tf.nn.relu(modelGraph['conv2_2'] + vbbConstants(7))
    modelGraph['pool2'] = tf.nn.max_pool(modelGraph['relu2_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    modelGraph['conv3_1'] = tf.nn.conv2d(modelGraph['pool2'], filter = vbbWeights(10), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu3_1'] = tf.nn.relu(modelGraph['conv3_1'] + vbbConstants(10))
    modelGraph['conv3_2'] = tf.nn.conv2d(modelGraph['relu3_1'], filter = vbbWeights(12), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu3_2'] = tf.nn.relu(modelGraph['conv3_2'] + vbbConstants(12))
    modelGraph['conv3_3'] = tf.nn.conv2d(modelGraph['relu3_2'], filter = vbbWeights(14), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu3_3'] = tf.nn.relu(modelGraph['conv3_3'] + vbbConstants(14))
    modelGraph['pool3'] = tf.nn.max_pool(modelGraph['relu3_3'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    modelGraph['conv4_1'] = tf.nn.conv2d(modelGraph['pool3'], filter = vbbWeights(17), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu4_1'] = tf.nn.relu(modelGraph['conv4_1'] + vbbConstants(17))
    modelGraph['conv4_2'] = tf.nn.conv2d(modelGraph['relu4_1'], filter = vbbWeights(19), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu4_2'] = tf.nn.relu(modelGraph['conv4_2'] + vbbConstants(19))
    modelGraph['conv4_3'] = tf.nn.conv2d(modelGraph['relu4_2'], filter = vbbWeights(21), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu4_3'] = tf.nn.relu(modelGraph['conv4_3'] + vbbConstants(21))
    modelGraph['pool4'] = tf.nn.max_pool(modelGraph['relu4_3'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    modelGraph['conv5_1'] = tf.nn.conv2d(modelGraph['pool4'], filter = vbbWeights(24), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu5_1'] = tf.nn.relu(modelGraph['conv5_1'] + vbbConstants(24))
    modelGraph['conv5_2'] = tf.nn.conv2d(modelGraph['relu5_1'], filter = vbbWeights(26), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu5_2'] = tf.nn.relu(modelGraph['conv5_2'] + vbbConstants(26))
    modelGraph['conv5_3'] = tf.nn.conv2d(modelGraph['relu5_2'], filter = vbbWeights(28), strides = [1, 1, 1, 1], padding = 'SAME')
    modelGraph['relu5_3'] = tf.nn.relu(modelGraph['conv5_3'] + vbbConstants(28))
    modelGraph['pool5'] = tf.nn.max_pool(modelGraph['relu5_3'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    modelGraph['resh1'] = tf.reshape(modelGraph['pool5'], [-1, 25088])
    modelGraph['fc6'] = tf.nn.relu_layer(modelGraph['resh1'], tf.reshape(vbbWeights(31), [25088, 4096]), vbbConstants(31))
    modelGraph['dropout1'] = tf.nn.dropout(modelGraph['fc6'], 0.5)
    modelGraph['fc7'] = tf.nn.relu_layer(modelGraph['dropout1'], tf.squeeze(vbbWeights(34), [0, 1]), vbbConstants(34))
    modelGraph['dropout2'] = tf.nn.dropout(modelGraph['fc7'], 0.5)
    modelGraph['fc8'] = tf.nn.relu_layer(modelGraph['dropout2'], tf.squeeze(vbbWeights(37), [0, 1]), vbbConstants(37))

    return modelGraph
