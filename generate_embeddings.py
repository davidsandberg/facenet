import facenet
import tensorflow as tf
import numpy as np
import imageio
import os
from tensorflow.python.platform import gfile
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def generate_embeddings(image_names, model_dir, chunk_size = 512):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_placeholder = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            if len(image_names)<chunk_size:
                images = np.asarray([prewhiten(imageio.imread(image_name)) for image_name in image_names])
                feed_dict = {images_placeholder:images, phase_train_placeholder:False}
                embeddings = sess.run(embeddings_placeholder, feed_dict=feed_dict)
            else:
                embeddings = np.empty((0, 128))
                for i in range(0, len(image_names), chunk_size):
                    image_names_chunk = image_names[i:min(i+chunk_size, len(image_names))]
                    images_chunk = np.asarray([prewhiten(imageio.imread(image_name)) for image_name in image_names_chunk])
                    feed_dict = {images_placeholder:images_chunk, phase_train_placeholder:False}
                    embedding_chunk = sess.run(embeddings_placeholder, feed_dict=feed_dict)
                    embeddings = np.vstack((embeddings,embedding_chunk))
            return embeddings

#CCNA training data evaluation
ccna_data = '/datadrive/images/ccna_cropped_data'
ccna_labels = os.listdir(ccna_data)
ccna_sku_full_paths = [ccna_data+'/'+sku for sku in ccna_labels]
ccna_embedding_dict = {}
ccna_sku_image_names_dict = {}
ccna_full_training_image_names_list = []
for sku in ccna_sku_full_paths:
    sku_image_names = [sku+'/'+f for f in os.listdir(sku)]
    ccna_sku_image_names_dict[sku.split('/')[-1]] = sku_image_names
    ccna_full_training_image_names_list += sku_image_names
full_training_embeddings = generate_embeddings(ccna_full_training_image_names_list, '/home/caffe/facenet/sku_triplet_500k.pb', chunk_size = 512)
np.save('CCNA_full_training_embeddings_facenet_model_v1.npy', full_training_embeddings)
np.save('ccna_full_training_image_names_list.npy', np.asarray(ccna_full_training_image_names_list))
