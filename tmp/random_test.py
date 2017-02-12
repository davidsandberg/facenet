import tensorflow as tf
import numpy as np



with tf.Graph().as_default():
  tf.set_random_seed(666)


  # Placeholder for input images
  input_placeholder = tf.placeholder(tf.float32, shape=(9, 7), name='input')
  
  # Split example embeddings into anchor, positive and negative
  #anchor, positive, negative = tf.split(0, 3, input)
  resh1 = tf.reshape(input_placeholder, [3,3,7])
  anchor = resh1[0,:,:]
  positive = resh1[1,:,:]
  negative = resh1[2,:,:]
  
  # Build an initialization operation to run below.
  init = tf.global_variables_initializer()

  # Start running operations on the Graph.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
  sess.run(init)
  
  with sess.as_default():
    batch = np.zeros((9,7))
    batch[0,:] = 1.1
    batch[1,:] = 2.1
    batch[2,:] = 3.1
    batch[3,:] = 1.2
    batch[4,:] = 2.2
    batch[5,:] = 3.2
    batch[6,:] = 1.3
    batch[7,:] = 2.3
    batch[8,:] = 3.3
    feed_dict = {input_placeholder: batch }
    print(batch)
    print(sess.run([anchor, positive, negative], feed_dict=feed_dict))




#feed_dict = { images_placeholder: np.zeros((90,96,96,3)), phase_train_placeholder: True }
#vars_eval  = sess.run(tf.global_variables(), feed_dict=feed_dict)
#for gt in vars_eval:
  #print('%.20f' % (np.sum(gt)))
#for gt, gv in zip(grads_eval, grad_vars):
  #print('%40s: %.20f' % (gv.op.name, np.sum(gt)))

  

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
#init = tf.global_variables_initializer()

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
