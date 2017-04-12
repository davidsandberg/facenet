import numpy as np
from scipy import misc
import tensorflow as tf
from matplotlib import pyplot, image
import vggverydeep19

paintingStyleImage = image.imread("../data/schoolofathens.jpg")
pyplot.imshow(paintingStyleImage)

inputImage = image.imread("../data/grandcentral.jpg")
pyplot.imshow(inputImage)

outputWidth = 800
outputHeight = 600

# Beta constant 
beta = 5
# Alpha constant
alpha = 100
# Noise ratio
noiseRatio = 0.6

nodes = vggverydeep19.load('../data/imagenet-vgg-verydeep-19.mat', (600, 800))

# Mean VGG-19 image
meanImage19 = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3)) #pylint: disable=no-member



# Squared-error loss of content between the two feature representations
def sqErrorLossContent(sess, modelGraph, layer):
    p = session.run(modelGraph[layer])
    #pylint: disable=maybe-no-member
    N = p.shape[3]
    M = p.shape[1] * p.shape[2]
    return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(modelGraph[layer] - sess.run(modelGraph[layer]), 2))
 
# Squared-error loss of style between the two feature representations
styleLayers = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2),
]
def sqErrorLossStyle(sess, modelGraph):
    def intermediateCalc(x, y):
        N = x.shape[3]
        M = x.shape[1] * x.shape[2]
        A = tf.matmul(tf.transpose(tf.reshape(x, (M, N))), tf.reshape(x, (M, N)))
        G = tf.matmul(tf.transpose(tf.reshape(y, (M, N))), tf.reshape(y, (M, N)))
        return (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
    E = [intermediateCalc(sess.run(modelGraph[layerName]), modelGraph[layerName]) for layerName, _ in styleLayers]
    W = [w for _, w in styleLayers]
    return sum([W[layerNumber] * E[layerNumber] for layerNumber in range(len(styleLayers))])

session = tf.InteractiveSession()
 
# Addition of extra dimension to image
inputImage = np.reshape(inputImage, ((1,) + inputImage.shape))
inputImage = inputImage - meanImage19
# Display image
pyplot.imshow(inputImage[0])

# Addition of extra dimension to image
paintingStyleImage = np.reshape(paintingStyleImage, ((1,) + paintingStyleImage.shape))
paintingStyleImage = paintingStyleImage - meanImage19
# Display image
pyplot.imshow(paintingStyleImage[0])

imageNoise = np.random.uniform(-20, 20, (1, outputHeight, outputWidth, 3)).astype('float32')
pyplot.imshow(imageNoise[0])
mixedImage = imageNoise * noiseRatio + inputImage * (1 - noiseRatio)
pyplot.imshow(inputImage[0])


session.run(tf.global_variables_initializer())
session.run(nodes['input'].assign(inputImage))
contentLoss = sqErrorLossContent(session, nodes, 'conv4_2')
session.run(nodes['input'].assign(paintingStyleImage))
styleLoss = sqErrorLossStyle(session, nodes)
totalLoss = beta * contentLoss + alpha * styleLoss

optimizer = tf.train.AdamOptimizer(2.0)
trainStep = optimizer.minimize(totalLoss)
session.run(tf.global_variables_initializer())
session.run(nodes['input'].assign(inputImage))
# Number of iterations to run.
iterations = 2000
session.run(tf.global_variables_initializer())
session.run(nodes['input'].assign(inputImage))
 
for iters in range(iterations):
    session.run(trainStep)
    if iters%50 == 0:
        # Output every 50 iterations for animation       
        filename = 'output%d.png' % (iters)
        im = mixedImage + meanImage19
        im = im[0]
        im = np.clip(im, 0, 255).astype('uint8')
        misc.imsave(filename, im)
 
im = mixedImage + meanImage19
im = im[0]
im = np.clip(im, 0, 255).astype('uint8')
misc.imsave('finalImage.png', im)

