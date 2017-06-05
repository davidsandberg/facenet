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
import os
import shutil
import numpy as np
import facenet
import math

class GetLearningRateFromFileTest(unittest.TestCase):
  
    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp()
        
    @classmethod
    def tearDownClass(self):
        # Recursively remove the temporary directory
        shutil.rmtree(self.tmp_dir)

    def test_stuff(self):
        schedule_path = os.path.join(self.tmp_dir, 'lr1.txt')
        schedule_dict = { 0: 0.001, 10:0.001, 20:0.0001}
        #schedule_dict = { 0: 0.05, 30:0.005, 150:0.0001, 200:0.0001}
        self.write_learning_rate_schedule_file(schedule_path, schedule_dict)
        self.assertTrue(math.isnan(facenet.get_learning_rate_from_file(schedule_path, -1)))
        self.assertAlmostEqual(facenet.get_learning_rate_from_file(schedule_path, 0), 0.001, 1e-6)
        self.assertAlmostEqual(facenet.get_learning_rate_from_file(schedule_path, 5), 0.001, 1e-6)
        self.assertAlmostEqual(facenet.get_learning_rate_from_file(schedule_path, 10), 0.001, 1e-6)
        self.assertAlmostEqual(facenet.get_learning_rate_from_file(schedule_path, 15), np.power(10.0, -3.5), 1e-6)
        self.assertAlmostEqual(facenet.get_learning_rate_from_file(schedule_path, 20), 0.0001, 1e-6)
        self.assertTrue(math.isnan(facenet.get_learning_rate_from_file(schedule_path, 21)))
#         array([ 0.001     ,  0.00079433,  0.00063096,  0.00050119,  0.00039811,
#         0.00031623,  0.00025119,  0.00019953,  0.00015849,  0.00012589,
#         0.0001    ])
        
    # Write a learning rate schedule file
    def write_learning_rate_schedule_file(self, filename, schedule_dict):
        with open(filename, 'w') as f:
            for epoch in schedule_dict:
                f.write('%d: %.4f\n' % (epoch, schedule_dict[epoch]))
        
if __name__ == "__main__":
    unittest.main()
    