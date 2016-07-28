import unittest
import facenet_train
import sys

class TrainTest(unittest.TestCase):


    def test_training(self):
        arg_string = '--logs_base_dir /media/david/BigDrive/DeepLearning/logs/facenet/ --data_dir ~/datasets/facescrub/facescrub_new_96_96 --epoch_size 58'
        args = facenet_train.parse_arguments(arg_string.split(' '))
        facenet_train.main(args)
        #testing.assert_almost_equal(y1, y2, 10, 'Output from two forward passes with phase_train==false should be equal')


if __name__ == "__main__":
    unittest.main()
    