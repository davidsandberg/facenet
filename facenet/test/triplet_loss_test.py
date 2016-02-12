import tensorflow as tf
import numpy as np
import facenet
import h5py    # HDF5 support

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('alpha', 0.2,
                            """Positive to negative triplet distance margin.""")


with tf.Graph().as_default():
  
  embeddings_placeholder = tf.placeholder(tf.float32, shape=(6, 10), name='Input')
  
  a, p, n = tf.split(0, 3, embeddings_placeholder)
  
  # Calculate triplet loss
  loss = facenet.triplet_loss(a, p, n)
  
  
  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()
  
  # Start running operations on the Graph.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
  sess.run(init)
  
  
  with sess.as_default():
    
    
    fileName = "/home/david/tripletLossTest3.h5"
    f = h5py.File(fileName,  "r")
    
    if True:
      embeddings = f['input']    
    else:
      embeddings = np.zeros((3,128))
      np.random.seed(123)
      for ix in range(embeddings.shape[0]):
        for jx in range(embeddings.shape[1]):
          rnd = 1.0*np.random.randint(1,2**32)/2**32
          embeddings[ix][jx] = rnd
        
    #for i in range(128):
      #print('%1.5f  %1.5f  %1.5f' % (embeddings[0,i], embeddings[1,i], embeddings[2,i], ))
    
    feed_dict = { embeddings_placeholder: embeddings }
    err  = sess.run([loss], feed_dict=feed_dict)
    
    refLoss = f['loss'][0,0]
    loss = err[0]
    print('refLoss: %3.12f  loss: %3.12f  diff: %3.12f' % (refLoss, loss, loss-refLoss))
    
          
    #input = f['input']
    #for i in range(128):
      #print('%1.12f  %1.12f  %1.12f' % (embeddings[0,i]-input[0,i], embeddings[1,i]-input[1,i], embeddings[2,i]-input[2,i] ))
