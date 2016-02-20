import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append('../src')
import facenet

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
  
    tf.set_random_seed(666)
    global_step = tf.Variable(0, trainable=False)
    
    # Placeholder for input images
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='Input')
    
    # Placeholder for phase_train
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    
    # Build the inference graph
    embeddings = facenet.inference_nn4_max_pool_96(images_placeholder, phase_train=phase_train_placeholder)
    
    # Split example embeddings into anchor, positive and negative
    anchor, positive, negative = tf.split(0, 3, embeddings)
    
    # Calculate triplet loss
    loss = facenet.triplet_loss(anchor, positive, negative)
    
    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_op, grads = facenet.train(loss, global_step)
    
    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()
    
    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    np.random.seed(666)
    
    with sess.as_default():
      grads_eval = []
      all_vars = []
      for step in xrange(1):
        batch = np.random.random((90,96,96,3))
        feed_dict = { images_placeholder: batch, phase_train_placeholder: True }
        grad_tensors, grad_vars = zip(*grads)
        var_names = tf.all_variables()
        all_vars  += sess.run(var_names, feed_dict=feed_dict)
        grads_eval  += sess.run(grad_tensors, feed_dict=feed_dict)
        sess.run(train_op, feed_dict=feed_dict)
    
    time.sleep(3)
    sess.close()
  return (var_names, all_vars, grad_vars, grads_eval)

var_names1, all_vars1, grad_names1, all_grads1 = run_train()
var_names2, all_vars2, grad_names2, all_grads2 = run_train()

all_vars_close = [None] * len(all_vars1)
for i in range(len(all_vars1)):
  all_vars_close[i] = np.allclose(all_vars1[i], all_vars2[i], rtol=1.e-24)
  print('%d var: %s'%(i, var_names1[i].op.name), all_vars_close[i])
xxx = 1
  
all_grads_close = [None] * len(all_grads1)
for i in range(len(all_grads1)):
  all_grads_close[i] = np.allclose(all_grads1[i], all_grads2[i], rtol=1.e-24)
  print('%d grad: %s'%(i, grad_names1[i].op.name), all_grads_close[i])

for i in range(len(all_grads_close)):
  if all_grads_close[i]==False:
    print(grad_names1[i].op.name)

print('All vars close: ', all(all_vars_close))
print('All grads close: ', all(all_grads_close))

xxx = 1
