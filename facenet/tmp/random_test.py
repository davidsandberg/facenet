#feed_dict = { images_placeholder: np.zeros((90,96,96,3)), phase_train_placeholder: True }
#vars_eval  = sess.run(tf.all_variables(), feed_dict=feed_dict)
#for gt in vars_eval:
  #print('%.20f' % (np.sum(gt)))
#for gt, gv in zip(grads_eval, grad_vars):
  #print('%40s: %.20f' % (gv.op.name, np.sum(gt)))



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
  all_vars_close[i] = np.allclose(all_vars1[i], all_vars2[i])
  print('%d var: %s'%(i, var_names1[i].op.name), all_vars_close[i])
print('All vars close: ', all(all_vars_close))
xxx = 1
  
all_grads_close = [None] * len(all_grads1)
for i in range(len(all_grads1)):
  all_grads_close[i] = np.allclose(all_grads1[i], all_grads2[i])
  print('%d grad: %s'%(i, grad_names1[i].op.name), all_grads_close[i])
print('All grads close: ', all(all_grads_close))
np.flatnonzero(all_grads_close==False)

for i in range(len(all_grads_close)):
  if all_grads_close[i]==False:
    print(grad_names1[i].op.name)

xxx = 1


#import tensorflow as tf



#with tf.Graph().as_default():
  #tf.set_random_seed(666)


  #kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 32],
                                           #dtype=tf.float32,
                                           #stddev=1e-1), name='weights')
  
  ## Build an initialization operation to run below.
  #init = tf.initialize_all_variables()

  ## Start running operations on the Graph.
  #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
  #sess.run(init)
  
  #with sess.as_default():

    #print(sess.run(kernel))
  

#import h5py
#myFile = h5py.File('/home/david/repo/TensorFace/network.h5', 'r')

## The '...' means retrieve the whole tensor
#data = myFile[...]
#print(data)


#import h5py    # HDF5 support

#fileName = "/home/david/repo/TensorFace/network.h5"
#f = h5py.File(fileName,  "r")
##for item in f.keys():
  ##print item
#for item in f.values():
  #print item


#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#import math
#import facenet
#import os
#import glob
#from scipy import misc

#def plot_triplet(apn, idx):
    #plt.subplot(1,3,1)
    #plt.imshow(np.multiply(apn[idx*3+0,:,:,:],1/256))
    #plt.subplot(1,3,2)
    #plt.imshow(np.multiply(apn[idx*3+1,:,:,:],1/256))
    #plt.subplot(1,3,3)
    #plt.imshow(np.multiply(apn[idx*3+2,:,:,:],1/256))


#input_image = tf.placeholder(tf.float32, name='input_image')
#phase_train = tf.placeholder(tf.bool, name='phase_train')

#n_in, n_out = 3, 16
#ksize = 3
#stride = 1
#kernel = tf.Variable(tf.truncated_normal([ksize, ksize, n_in, n_out],
                                         #stddev=math.sqrt(2/(ksize*ksize*n_out))),
                     #name='kernel')
#conv = tf.nn.conv2d(input_image, kernel, [1,stride,stride,1], padding="SAME")
#conv_bn = facenet.batch_norm(conv, n_out, phase_train)
#relu = tf.nn.relu(conv_bn)

## Build an initialization operation to run below.
#init = tf.initialize_all_variables()

## Start running operations on the Graph.
#sess = tf.Session()
#sess.run(init)

#path = '/home/david/datasets/fs_aligned/Zooey_Deschanel/'
#files = glob.glob(os.path.join(path, '*.png'))
#nrof_samples = 30
#img_list = [None] * nrof_samples
#for i in xrange(nrof_samples):
    #img_list[i] = misc.imread(files[i])
#images = np.stack(img_list)

#feed_dict = {
    #input_image: images.astype(np.float32),
    #phase_train: True
#}

#out = sess.run([relu], feed_dict=feed_dict)
#print(out[0].shape)

##print(out)

#plot_triplet(images, 0)



#import matplotlib.pyplot as plt
#import numpy as np

#a=[3,4,5,6]
#b = [1,a[1:3]]
#print(b)

## Generate some data...
#x, y = np.meshgrid(np.linspace(-2,2,200), np.linspace(-2,2,200))
#x, y = x - x.mean(), y - y.mean()
#z = x * np.exp(-x**2 - y**2)
#print(z.shape)

## Plot the grid
#plt.imshow(z)
#plt.gray()
#plt.show()

#import numpy as np

#np.random.seed(123)
#rnd = 1.0*np.random.randint(1,2**32)/2**32
#print(rnd)
