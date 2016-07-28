import unittest
import facenet_train
import tempfile


class TrainTest(unittest.TestCase):

    def test_training_nn4(self):
        tmp_dir = tempfile.mkdtemp()
        argv = ['--logs_base_dir', tmp_dir,
                '--models_base_dir', tmp_dir,
                '--data_dir', '~/datasets/facescrub/facescrub_new_96_96',
                '--epoch_size', '1',
                '--max_nrof_epochs', '1',
                '--batch_size', '6',
                '--people_per_batch', '2',
                '--images_per_person', '3']
        args = facenet_train.parse_arguments(argv)
        facenet_train.main(args)
        #testing.assert_almost_equal(y1, y2, 10, 'Output from two forward passes with phase_train==false should be equal')


if __name__ == "__main__":
    unittest.main()
    