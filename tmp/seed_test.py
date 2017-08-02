import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append('../src')
import facenet
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

from six.moves import xrange

tf.app.flags.DEFINE_integer('batch_size', 90,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_float('alpha', 0.2,
                          """Positive to negative triplet distance margin.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """Expontential decay for tracking of training parameters.""")

FLAGS = tf.app.flags.FLAGS

def run_train():
  
  with tf.Graph().as_default():
  
    # Set the seed for the graph
    tf.set_random_seed(666)

    # Placeholder for input images
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')
    
    # Build the inference graph
    embeddings = inference_conv_test(images_placeholder)
    #embeddings = inference_affine_test(images_placeholder)
    
    # Split example embeddings into anchor, positive and negative
    anchor, positive, negative = tf.split(0, 3, embeddings)

    # Alternative implementation of the split operation
    # This produces the same error
    #resh1 = tf.reshape(embeddings, [3,int(FLAGS.batch_size/3), 128])
    #anchor = resh1[0,:,:]
    #positive = resh1[1,:,:]
    #negative = resh1[2,:,:]
    
    # Calculate triplet loss
    pos_dist = tf.reduce_sum(tf.square(tf.sub(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.sub(anchor, negative)), 1)
    basic_loss = tf.add(tf.sub(pos_dist,neg_dist), FLAGS.alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    #opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)  # Optimizer does not seem to matter
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads)
    
    # Initialize the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    # Set the numpy seed
    np.random.seed(666)
    
    with sess.as_default():
      grads_eval = []
      all_vars = []
      for step in xrange(1):
        # Generate some random input data
        batch = np.random.random((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))
        feed_dict = { images_placeholder: batch }
        # Get the variables
        var_names = tf.global_variables()
        all_vars  += sess.run(var_names, feed_dict=feed_dict)
        # Get the gradients
        grad_tensors, grad_vars = zip(*grads)
        grads_eval  += sess.run(grad_tensors, feed_dict=feed_dict)
        # Run training
        sess.run(train_op, feed_dict=feed_dict)
    
    sess.close()
  return (var_names, all_vars, grad_vars, grads_eval)

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
  kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(inpOp, kernel, [1, dH, dW, 1], padding=padType)
  
  biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
  conv1 = tf.nn.relu(bias)
  return conv1

def _affine(inpOp, nIn, nOut):
  kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                       trainable=True, name='biases')
  affine1 = tf.nn.relu_layer(inpOp, kernel, biases)
  return affine1
  
def inference_conv_test(images):
  conv1 = _conv(images, 3, 64, 7, 7, 2, 2, 'SAME')
  resh1 = tf.reshape(conv1, [-1, 147456])
  affn = _affine(resh1, 147456, 128)  # Affine layer not needed to reproduce the error
  return affn

def inference_affine_test(images):
  resh1 = tf.reshape(images, [-1, 27648])
  affn1 = _affine(resh1, 27648, 1024)
  affn2 = _affine(affn1, 1024, 1024)
  affn3 = _affine(affn2, 1024, 1024)
  affn4 = _affine(affn3, 1024, 128)
  return affn4

# Run two sessions with the same seed. These runs should produce the same result.
var_names1, all_vars1, grad_names1, all_grads1 = run_train()
var_names2, all_vars2, grad_names2, all_grads2 = run_train()

all_vars_close = [None] * len(all_vars1)
for i in range(len(all_vars1)):
  all_vars_close[i] = np.allclose(all_vars1[i], all_vars2[i], rtol=1.e-16)
  print('%d var %s: %s' % (i, var_names1[i].op.name, all_vars_close[i]))
  
all_grads_close = [None] * len(all_grads1)
for i in range(len(all_grads1)):
  all_grads_close[i] = np.allclose(all_grads1[i], all_grads2[i], rtol=1.e-16)
  print('%d grad %s: %s' % (i, grad_names1[i].op.name, all_grads_close[i]))

assert all(all_vars_close), 'Variable values differ between the two sessions (with the same seed)'
assert all(all_grads_close), 'Gradient values differ between the two sessions (with the same seed)'
