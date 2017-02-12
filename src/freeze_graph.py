"""Imports a model metagraph and checkpoint file, converts the variables to constants
and exports the model as a graphdef protobuf
"""
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
<<<<<<< HEAD
            saver = tf.train.import_meta_graph(os.path.join(os.path.expanduser(args.model_dir), 
                'model-' + os.path.basename(os.path.normpath(args.model_dir)) + '.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(args.model_dir)))
=======
            #meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            meta_file = os.path.join(os.path.expanduser(args.model_dir),'model-20161231-150622.meta')
            ckpt_file = os.path.join(os.path.expanduser(args.model_dir),'model-20161231-150622.ckpt-80000')
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
>>>>>>> refs/remotes/origin/dataset_filtering
            output_node_names = 'embeddings'
            whitelist_names = []
            for node in sess.graph.as_graph_def().node:
                if node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or node.name.startswith('phase_train'):
                    print(node.name)
                    whitelist_names.append(node.name)

            output_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), output_node_names.split(","), 
                variable_names_whitelist=whitelist_names)
        
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node)) #pylint: disable=no-member
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
