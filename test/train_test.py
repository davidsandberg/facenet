# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
import tempfile
import numpy as np
import cv2
import os
import shutil
import tensorflow as tf
import facenet_train
import facenet_train_classifier
import validate_on_lfw
import compare
import visualize
import test_invariance_on_lfw
import download_and_extract_model

class TrainTest(unittest.TestCase):
  
    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.tmp_dir, 'dataset')
        create_mock_dataset(self.dataset_dir)
        self.lfw_pairs_file = create_mock_lfw_pairs(self.tmp_dir)
        print(self.lfw_pairs_file)
        self.pretrained_model_name = '20170131-234652'
        download_and_extract_model.download_and_extract_model(self.pretrained_model_name, 'data/')
        
    @classmethod
    def tearDownClass(self):
        # Recursively remove the temporary directory
        shutil.rmtree(self.tmp_dir)

    @unittest.skip("Skip this test case for now")
    def test_training_nn4(self):
        argv = ['--logs_base_dir', self.tmp_dir,
                '--models_base_dir', self.tmp_dir,
                '--data_dir', self.dataset_dir,
                '--model_def', 'models.nn4',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '6',
                '--people_per_batch', '2',
                '--images_per_person', '3',
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2' ]
        args = facenet_train.parse_arguments(argv)
        model_dir = facenet_train.main(args)
        
        
        model_file = os.path.join(model_dir, 'model.ckpt-1')
        # Check that the trained model can be loaded
        tf.reset_default_graph()
        argv = [model_file,
                self.dataset_dir,
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_nrof_folds', '2' ]
        args = validate_on_lfw.parse_arguments(argv)
        validate_on_lfw.main(args)
        
    # test_align_dataset_mtcnn
    # http://vis-www.cs.umass.edu/lfw/lfw-a.zip
    
    # test_validate_on_lfw
    
    # test_triplet_loss_training
    
    # test_freeze_graph

    @unittest.skip("Skip this test case for now")
    def test_training_nn4_small2_v1(self):
        argv = ['--logs_base_dir', self.tmp_dir,
                '--models_base_dir', self.tmp_dir,
                '--data_dir', self.dataset_dir,
                '--model_def', 'models.nn4_small2_v1',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '6',
                '--people_per_batch', '2',
                '--images_per_person', '3',
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2' ]
        args = facenet_train.parse_arguments(argv)
        facenet_train.main(args)

    @unittest.skip("Skip this test case for now")
    def test_training_classifier_nn4(self):
        argv = ['--logs_base_dir', self.tmp_dir,
                '--models_base_dir', self.tmp_dir,
                '--data_dir', self.dataset_dir,
                '--model_def', 'models.nn4',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '6',
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2' ]
        args = facenet_train_classifier.parse_arguments(argv)
        facenet_train_classifier.main(args)

    def test_training_classifier_inception_resnet_v1(self):
        argv = ['--logs_base_dir', self.tmp_dir,
                '--models_base_dir', self.tmp_dir,
                '--data_dir', self.dataset_dir,
                '--model_def', 'models.inception_resnet_v1',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '1',
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2',
                '--lfw_batch_size', '1',
                '--nrof_preprocess_threads', '1' ]
        args = facenet_train_classifier.parse_arguments(argv)
        facenet_train_classifier.main(args)

    def test_training_classifier_inception_resnet_v2(self):
        argv = ['--logs_base_dir', self.tmp_dir,
                '--models_base_dir', self.tmp_dir,
                '--data_dir', self.dataset_dir,
                '--model_def', 'models.inception_resnet_v2',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '1',
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2',
                '--lfw_batch_size', '1',
                '--nrof_preprocess_threads', '1' ]
        args = facenet_train_classifier.parse_arguments(argv)
        facenet_train_classifier.main(args)

    def test_compare(self):
        argv = [os.path.join('data/', self.pretrained_model_name),
                'data/images/Anthony_Hopkins_0001.jpg',
                'data/images/Anthony_Hopkins_0002.jpg' ]
        args = compare.parse_arguments(argv)
        compare.main(args)

    @unittest.skip("Skip this test case for now")
    def test_visualize(self):
        model_dir = os.path.abspath('../data/model/20160620-173927')
        create_checkpoint_file(model_dir, 'model.ckpt-500000')
        argv = [model_dir, 
                '--model_def', 'models.nn4' ]
        args = visualize.parse_arguments(argv)
        visualize.main(args)

    @unittest.skip("Skip this test case for now")
    def test_test_invariance_on_lfw(self):
        model_dir = os.path.abspath('../data/model/20160620-173927')
        model_file = os.path.join(model_dir, 'model.ckpt-500000')
        argv = ['--model_file', model_file,
                '--lfw_pairs', self.lfw_pairs_file,
                '--lfw_dir', self.dataset_dir,
                '--lfw_nrof_folds', '2',
                '--orig_image_size', '96',
                '--nrof_offsets', '1',
                '--nrof_angles', '1',
                '--nrof_scales', '1' ]
        args = test_invariance_on_lfw.parse_arguments(argv)
        test_invariance_on_lfw.main(args)

# Create a checkpoint file pointing to the model
def create_checkpoint_file(model_dir, model_file):
    checkpoint_filename = os.path.join(model_dir, 'checkpoint')
    full_model_filename = os.path.join(model_dir, model_file)
    with open(checkpoint_filename, 'w') as f:
        f.write('model_checkpoint_path: "%s"\n' % full_model_filename)
        f.write('all_model_checkpoint_paths: "%s"\n' % full_model_filename)
        
# Create a mock dataset with random pixel images
def create_mock_dataset(dataset_dir):
   
    nrof_persons = 3
    nrof_images_per_person = 2
    np.random.seed(seed=666)
    os.mkdir(dataset_dir)
    for i in range(nrof_persons):
        class_name = '%04d' % (i+1)
        class_dir = os.path.join(dataset_dir, class_name)
        os.mkdir(class_dir)
        for j in range(nrof_images_per_person):
            img_name = '%04d' % (j+1)
            img_path = os.path.join(class_dir, class_name+'_'+img_name + '.png')
            img = np.random.uniform(low=0.0, high=255.0, size=(96,96,3))
            cv2.imwrite(img_path, img) #@UndefinedVariable

# Create a mock LFW pairs file
def create_mock_lfw_pairs(tmp_dir):
    pairs_filename = os.path.join(tmp_dir, 'pairs_mock.txt')
    with open(pairs_filename, 'w') as f:
        f.write('10 300\n')
        f.write('0001 1 2\n')
        f.write('0001 1 0002 1\n')
        f.write('0002 1 0003 1\n')
        f.write('0001 1 0003 1\n')
        f.write('0002 1 2\n')
        f.write('0001 2 0002 2\n')
        f.write('0002 2 0003 2\n')
        f.write('0001 2 0003 2\n')
        f.write('0003 1 2\n')
        f.write('0001 1 0002 2\n')
        f.write('0002 1 0003 2\n')
        f.write('0001 1 0003 2\n')
    return pairs_filename

if __name__ == "__main__":
    unittest.main()
    