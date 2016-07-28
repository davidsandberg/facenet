import unittest
import facenet_train
import tempfile
import numpy as np
import cv2
import os
import shutil

class TrainTest(unittest.TestCase):
  
    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.tmp_dir, 'dataset')
        create_mock_dataset(self.dataset_dir)
        self.lfw_pairs_file = create_mock_lfw_pairs(self.tmp_dir)
        print(self.lfw_pairs_file)
        
    @classmethod
    def tearDownClass(self):
        # Recursively remove the temporary directory
        shutil.rmtree(self.tmp_dir)

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
        facenet_train.main(args)

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
            cv2.imwrite(img_path, img)

# Create a mock LFW pairs file
def create_mock_lfw_pairs(tmp_dir):
    pairs_filename = os.path.join(tmp_dir, 'pairs_mock.txt')
    with open(pairs_filename, 'w') as f:
        f.write('10 300\n')
        f.write('0001 1 2\n')
        f.write('0002 1 2\n')
        f.write('0003 1 2\n')
        f.write('0001 1 0002 1\n')
        f.write('0002 1 0003 1\n')
        f.write('0001 1 0003 1\n')
        f.write('0001 2 0002 2\n')
        f.write('0002 2 0003 2\n')
        f.write('0001 2 0003 2\n')
        f.write('0001 1 0002 2\n')
        f.write('0002 1 0003 2\n')
        f.write('0001 1 0003 2\n')
    return pairs_filename

if __name__ == "__main__":
    unittest.main()
    