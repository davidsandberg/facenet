"""Test invariance to translation, scaling and rotation on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
This requires test images to be cropped a bit wider than the normal to give some room for the transformations.
"""
import tensorflow as tf
import numpy as np
import facenet
import matplotlib.pyplot as plt
from scipy import misc
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_file', '~/models/facenet/20160514-234418/model.ckpt-500000',
                           """File containing the model parameters as well as the model metagraph (with extension '.meta')""")
tf.app.flags.DEFINE_string('lfw_pairs', '~/repo/facenet/data/lfw/pairs.txt',
                           """The file containing the pairs to use for validation.""")
tf.app.flags.DEFINE_string('file_ext', '.png',
                           """The file extension for the LFW dataset, typically .png or .jpg.""")
tf.app.flags.DEFINE_string('lfw_dir', '~/datasets/lfw/lfw_aligned_224/',
                           """Path to the data directory containing aligned face patches, cropped to give some room for image transformations.""")
tf.app.flags.DEFINE_integer('batch_size', 60,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('orig_image_size', 224,
                            """Image size (height, width) in pixels of the original (uncropped/unscaled) images.""")
tf.app.flags.DEFINE_integer('seed', 666, """Random seed.""")

def main(argv=None):
    
    pairs = read_pairs(os.path.expanduser(FLAGS.lfw_pairs))
    paths, actual_issame = get_paths(os.path.expanduser(FLAGS.lfw_dir), pairs)
    
    with tf.Graph().as_default():

        with tf.Session() as sess:
    
            # Load the model
            print('Loading model "%s"' % FLAGS.model_file)
            facenet.load_model(FLAGS.model_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            image_size = int(images_placeholder.get_shape()[1])
            
            # Run test on LFW to check accuracy for different horizontal/vertical translations of input images
            if True:
                offsets = np.arange(-30, 31, 3)
                horizontal_offset_accuracy = [None] * len(offsets)
                for idx, offset in enumerate(offsets):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, paths, actual_issame, translate_images, (offset,0))
                    print('Hoffset: %1.3f  Accuracy: %1.3f+-%1.3f' % (offset, np.mean(accuracy), np.std(accuracy)))
                    horizontal_offset_accuracy[idx] = np.mean(accuracy)
                vertical_offset_accuracy = [None] * len(offsets)
                for idx, offset in enumerate(offsets):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, paths, actual_issame, translate_images, (0,offset))
                    print('Voffset: %1.3f  Accuracy: %1.3f+-%1.3f' % (offset, np.mean(accuracy), np.std(accuracy)))
                    vertical_offset_accuracy[idx] = np.mean(accuracy)
                fig = plt.figure(1)
                plt.plot(offsets, horizontal_offset_accuracy, label='Horizontal')
                plt.plot(offsets, vertical_offset_accuracy, label='Vertical')
                plt.legend()
                plt.grid(True)
                plt.title('Translation invariance on LFW')
                plt.xlabel('Offset [pixels]')
                plt.ylabel('Accuracy')
                plt.show()
                result_dir = os.path.expanduser(FLAGS.model_dir)
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_translation.png'))
                save_result(offsets, horizontal_offset_accuracy, os.path.join(result_dir, 'invariance_translation_horizontal.txt'))
                save_result(offsets, vertical_offset_accuracy, os.path.join(result_dir, 'invariance_translation_vertical.txt'))

            # Run test on LFW to check accuracy for different rotation of input images
            if True:
                angles = np.arange(-30, 31, 3)
                rotation_accuracy = [None] * len(angles)
                for idx, angle in enumerate(angles):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, paths, actual_issame, rotate_images, angle)
                    print('Angle: %1.3f  Accuracy: %1.3f+-%1.3f' % (angle, np.mean(accuracy), np.std(accuracy)))
                    rotation_accuracy[idx] = np.mean(accuracy)
                fig = plt.figure(2)
                plt.plot(angles, rotation_accuracy)
                plt.grid(True)
                plt.title('Rotation invariance on LFW')
                plt.xlabel('Angle [deg]')
                plt.ylabel('Accuracy')
                plt.show()
                result_dir = os.path.expanduser(FLAGS.model_dir)
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_rotation.png'))
                save_result(angles, rotation_accuracy, os.path.join(result_dir, 'invariance_rotation.txt'))

            # Run test on LFW to check accuracy for different scaling of input images
            if True:
                scales = np.arange(0.5, 1.5, 0.05)
                scale_accuracy = [None] * len(scales)
                for scale_idx, scale in enumerate(scales):
                    accuracy = evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, paths, actual_issame, scale_images, scale)
                    print('Scale: %1.3f  Accuracy: %1.3f+-%1.3f' % (scale, np.mean(accuracy), np.std(accuracy)))
                    scale_accuracy[scale_idx] = np.mean(accuracy)
                fig = plt.figure(3)
                plt.plot(scales, scale_accuracy)
                plt.grid(True)
                plt.title('Scale invariance on LFW')
                plt.xlabel('Scale')
                plt.ylabel('Accuracy')
                plt.show()
                result_dir = os.path.expanduser(FLAGS.model_dir)
                print('Saving results in %s' % result_dir)
                fig.savefig(os.path.join(result_dir, 'invariance_scale.png'))
                save_result(scales, scale_accuracy, os.path.join(result_dir, 'invariance_scale.txt'))
                
def save_result(aug, acc, filename):
    with open(filename, "w") as f:
        for i in range(aug.size):
            f.write('%6.4f %6.4f\n' % (aug[i], acc[i]))
            
def evaluate_accuracy(sess, images_placeholder, phase_train_placeholder, image_size, embeddings, paths, actual_issame, augument_images, aug_value):
    nrof_images = len(paths)
    nrof_batches = int(nrof_images / FLAGS.batch_size)  # Run forward pass on the remainder in the last batch
    emb_list = []
    for i in range(nrof_batches):
        paths_batch = paths[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
        images = facenet.load_data(paths_batch, False, False, FLAGS.orig_image_size)
        images_aug = augument_images(images, aug_value, image_size)
        feed_dict = { images_placeholder: images_aug, phase_train_placeholder: False }
        emb_list += sess.run([embeddings], feed_dict=feed_dict)
    emb_array = np.vstack(emb_list)  # Stack the embeddings to a nrof_examples_per_epoch x 128 matrix
    
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]
    _, _, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), FLAGS.seed)
    return accuracy

def scale_images(images, scale, image_size):
    images_scale_list = [None] * images.shape[0]
    for i in range(images.shape[0]):
        images_scale_list[i] = misc.imresize(images[i,:,:,:], scale)
    images_scale = np.stack(images_scale_list,axis=0)
    sz1 = images_scale.shape[1]/2
    sz2 = image_size/2
    images_crop = images_scale[:,(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
    return images_crop

def rotate_images(images, angle, image_size):
    images_list = [None] * images.shape[0]
    for i in range(images.shape[0]):
        images_list[i] = misc.imrotate(images[i,:,:,:], angle)
    images_rot = np.stack(images_list,axis=0)
    sz1 = images_rot.shape[1]/2
    sz2 = image_size/2
    images_crop = images_rot[:,(sz1-sz2):(sz1+sz2),(sz1-sz2):(sz1+sz2),:]
    return images_crop

def translate_images(images, offset, image_size):
    h, v = offset
    sz1 = images.shape[1]/2
    sz2 = image_size/2
    images_crop = images[:,(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return images_crop

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+FLAGS.file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+FLAGS.file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+FLAGS.file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+FLAGS.file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    assert(len(pairs) == 6000)
    return np.array(pairs)

if __name__ == '__main__':
    tf.app.run()
