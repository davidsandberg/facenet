import unittest
import tempfile
import os
import shutil
import tensorflow as tf
import numpy as np

class TrainTest(unittest.TestCase):
  
    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(self):
        # Recursively remove the temporary directory
        shutil.rmtree(self.tmp_dir)

    def test_restore_noema(self):
        
        # Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
        x_data = np.random.rand(100).astype(np.float32)
        y_data = x_data * 0.1 + 0.3
        
        # Try to find values for W and b that compute y_data = W * x_data + b
        # (We know that W should be 0.1 and b 0.3, but TensorFlow will
        # figure that out for us.)
        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
        b = tf.Variable(tf.zeros([1]), name='b')
        y = W * x_data + b
        
        # Minimize the mean squared errors.
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)
        
        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        saver = tf.train.Saver(tf.trainable_variables())
        
        # Launch the graph.
        sess = tf.Session()
        sess.run(init)
        
        # Fit the line.
        for _ in range(201):
            sess.run(train)
        
        w_reference = sess.run('W:0')
        b_reference = sess.run('b:0')
        
        saver.save(sess, os.path.join(self.tmp_dir, "model_ex1"))
        
        tf.reset_default_graph()

        saver = tf.train.import_meta_graph(os.path.join(self.tmp_dir, "model_ex1.meta"))
        sess = tf.Session()
        saver.restore(sess, os.path.join(self.tmp_dir, "model_ex1"))
        
        w_restored = sess.run('W:0')
        b_restored = sess.run('b:0')
        
        self.assertAlmostEqual(w_reference, w_restored, 'Restored model use different weight than the original model')
        self.assertAlmostEqual(b_reference, b_restored, 'Restored model use different weight than the original model')


    def test_restore_ema(self):
        
        # Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
        x_data = np.random.rand(100).astype(np.float32)
        y_data = x_data * 0.1 + 0.3
        
        # Try to find values for W and b that compute y_data = W * x_data + b
        # (We know that W should be 0.1 and b 0.3, but TensorFlow will
        # figure that out for us.)
        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
        b = tf.Variable(tf.zeros([1]), name='b')
        y = W * x_data + b
        
        # Minimize the mean squared errors.
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        opt_op = optimizer.minimize(loss)

        # Track the moving averages of all trainable variables.
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        averages_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([opt_op]):
            train_op = tf.group(averages_op)
  
        # Before starting, initialize the variables.  We will 'run' this first.
        init = tf.initialize_all_variables()

        saver = tf.train.Saver(tf.trainable_variables())
        
        # Launch the graph.
        sess = tf.Session()
        sess.run(init)
        
        # Fit the line.
        for _ in range(201):
            sess.run(train_op)
        
        w_reference = sess.run('W/ExponentialMovingAverage:0')
        b_reference = sess.run('b/ExponentialMovingAverage:0')
        
        saver.save(sess, os.path.join(self.tmp_dir, "model_ex1"))
                
        tf.reset_default_graph()

        tf.train.import_meta_graph(os.path.join(self.tmp_dir, "model_ex1.meta"))
        sess = tf.Session()
        
        print('------------------------------------------------------')
        for var in tf.all_variables():
            print('all variables: ' + var.op.name)
        for var in tf.trainable_variables():
            print('normal variable: ' + var.op.name)
        for var in tf.moving_average_variables():
            print('ema variable: ' + var.op.name)
        print('------------------------------------------------------')

        mode = 1
        restore_vars = {}
        if mode == 0:
            ema = tf.train.ExponentialMovingAverage(1.0)
            for var in tf.trainable_variables():
                print('%s: %s' % (ema.average_name(var), var.op.name))
                restore_vars[ema.average_name(var)] = var
        elif mode == 1:
            for var in tf.trainable_variables():
                ema_name = var.op.name + '/ExponentialMovingAverage'
                print('%s: %s' % (ema_name, var.op.name))
                restore_vars[ema_name] = var
            
        saver = tf.train.Saver(restore_vars, name='ema_restore')
        
        saver.restore(sess, os.path.join(self.tmp_dir, "model_ex1"))
        
        w_restored = sess.run('W:0')
        b_restored = sess.run('b:0')
        
        self.assertAlmostEqual(w_reference, w_restored, 'Restored model modes not use the EMA filtered weight')
        self.assertAlmostEqual(b_reference, b_restored, 'Restored model modes not use the EMA filtered bias')

        
# Create a checkpoint file pointing to the model
def create_checkpoint_file(model_dir, model_file):
    checkpoint_filename = os.path.join(model_dir, 'checkpoint')
    full_model_filename = os.path.join(model_dir, model_file)
    with open(checkpoint_filename, 'w') as f:
        f.write('model_checkpoint_path: "%s"\n' % full_model_filename)
        f.write('all_model_checkpoint_paths: "%s"\n' % full_model_filename)
        
if __name__ == "__main__":
    unittest.main()
    