import unittest
import tensorflow as tf
import facenet
import numpy as np
import numpy.testing as testing
import facenet_train

class TrainTest(unittest.TestCase):


    def test_training(self):
      
        facenet_train.__main__('hej')
        #testing.assert_almost_equal(y1, y2, 10, 'Output from two forward passes with phase_train==false should be equal')


if __name__ == "__main__":
    unittest.main()
    