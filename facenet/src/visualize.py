"""Visualize higher layer filters by finding the input image that maximizes the response of a given filter using gradient ascent.
NOTE. The results are somewhat strange and the code needs to be verified.
"""
import tensorflow as tf
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '~/models/facenet/20160514-234418',
                           """Directory containing the graph definition and checkpoint files.""")
tf.app.flags.DEFINE_string('model_def', 'models.nn4',
                           """Model definition. Points to a module containing the definition of the inference graph.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 96,
                            """Image size (height, width) in pixels.""")
tf.app.flags.DEFINE_string('pool_type', 'MAX',
                          """The type of pooling to use for some of the inception layers {'MAX', 'L2'}.""")
tf.app.flags.DEFINE_boolean('use_lrn', False,
                          """Enables Local Response Normalization after the first layers of the inception network.""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """Random seed.""")

network = importlib.import_module(FLAGS.model_def, 'inference')

def main():
    # Set random seed
    np.random.seed(seed=FLAGS.seed)

    with tf.Graph().as_default():

        # Placeholder for input images
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='input')

        # Create a variable that enables us to calculate the gradient for the input image     
        input_grad = tf.Variable(tf.constant(0.0, tf.float32, (FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3)), name='input_grad')
        images = tf.add(images_placeholder, input_grad)
          
        # Placeholder for phase_train
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # Build the inference graph
        _ = network.inference(images, FLAGS.pool_type, FLAGS.use_lrn, 1.0, phase_train=phase_train_placeholder)
          
        # Get the tensor that contains the filters that we want to do gradient ascent on
        opt_tensor = tf.get_default_graph().get_tensor_by_name('incept5b/incept5b:0')

        # Create a loss function for the filter
        filter_index = 0
        opt_filter = tf.slice(opt_tensor, [0, filter_index, 0, 0], [-1, 1, 1, -1], 'slice')
        loss = tf.reduce_mean(opt_filter, None, False, 'loss')
        
        # Compute gradients
        with tf.control_dependencies([loss]):
            opt = tf.train.AdagradOptimizer(1.0)
            grads = opt.compute_gradients(loss)

        # Create a saver for restoring variable averages
        ema = tf.train.ExponentialMovingAverage(1.0)
        restore_vars = ema.variables_to_restore()
        # The gradient ascent variable is not included in the stored model. Remove from the restore dict.
        for t in ema.variables_to_restore():
            if 'input_grad' in t:
                del restore_vars[t]
        saver = tf.train.Saver(restore_vars)

        with tf.Session() as sess:
      
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(FLAGS.model_dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('Checkpoint not found')

            # Initialize the gradient ascent variable
            sess.run(input_grad.initializer)
    
            image = np.random.random((1, 96, 96, 3)) * 20.0 + 128.0
            nrof_iter = 1000
            for i in range(nrof_iter):
                feed_dict = { images_placeholder: image, phase_train_placeholder: False }
                grad_tensors, _ = zip(*grads)
                loss_eval, img_grad_eval  = sess.run((loss, grad_tensors[0]), feed_dict=feed_dict)
                norm_grad = normalize(img_grad_eval)
                image = np.add(image, norm_grad)
                print('iter: %4d  loss: %3.8f' % (i, loss_eval))

            plt.figure(1)
            final_image = scale_and_clip(image[0,:,:,:])
            plt.imshow(final_image)
            plt.show()

def normalize(x):
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)

def scale_and_clip(x):
    x_norm = (x-x.mean()) / (x.std() + 1e-6) * 0.1
    x_clip = np.clip(x_norm+0.5, 0, 1)
    return x_clip

if __name__ == '__main__':
    main()
