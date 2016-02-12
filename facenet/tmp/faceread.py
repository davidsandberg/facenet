import tensorflow as tf
import os
import glob
from os import path

def read_image_data():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    #Create a list of filenames
    #path = '/home/david/datasets/fs_ready/Aaron_Eckhart/'
    jpeg_files = glob.glob(os.path.join(path, '*.jpg'))
    path = '/home/david/datasets/fs_ready/Zooey_Deschanel/'
    #Create a queue that produces the filenames to read
    filename_queue = tf.train.string_input_producer(jpeg_files)
    #Create a reader for the filequeue
    reader = tf.WholeFileReader()
    #Read in the files
    key, value = reader.read(filename_queue)
    #Convert the Tensor(of type string) to representing the Tensor of type uint8
    # and shape [height, width, channels] representing the images
    images = tf.image.decode_jpeg(value, channels=3)
    #convert images to floats and attach image summary
    float_images = tf.expand_dims(tf.cast(images, tf.float32),0)
    tf.image_summary('images', float_images)
    
    #Create session
    sess = tf.Session()
    summary_op = tf.merge_all_summaries()
    tf.initialize_all_variables()
    #Write summary
    summary_writer = tf.train.SummaryWriter(dirname+'/log/', graph_def=sess.graph_def)
    tf.train.start_queue_runners(sess=sess)
    for i in xrange(10):
        summary_str, float_image = sess.run([summary_op, float_images])
        print (float_image.shape)
        summary_writer.add_summary(summary_str)
    #Close session
    sess.close()


def main(argv=None):
    read_image_data()
    return 0

if __name__ == '__main__':
    tf.app.run()