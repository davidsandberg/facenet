# -*- coding:utf-8 -*-
from __future__ import print_function

import argparse
import os
import random
import sys

import numpy as np
from tinytag import TinyTag


def gen_balanced(voxceleb2_dir, output_dir, max_per_class=-1e5):
    output_list = output_dir + 'train_balance.txt'
    id_name_list = output_dir + 'id_name_balance.csv'
    voxdirlen = len(voxceleb2_dir) + 1
    id_list = os.listdir(voxceleb2_dir)

    # files
    idx = 0
    id_count = []
    list_file = open(output_list, 'w')
    for id in id_list:
        subdir = voxceleb2_dir + '/' + id
        seqs = os.listdir(subdir)
        seq_list = []
        for seq in seqs:
            seq_dir = subdir + '/' + seq
            files = os.listdir(seq_dir)
            seq_pass = []
            for fpath in files:
                ftitle, fext = os.path.splitext(fpath)
                # images
                if fext.lower() in ['.jpg', '.png', '.bmp', '.jpeg']:
                    rel_path = id + '/' + seq + '/' + fpath
                    seq_pass.append(rel_path)
            seq_list.append(seq_pass)
        # each seq
        valid = 0
        seq_prob = float(max_per_class) / len(seq_list)
        for list in seq_list:
            item_prob = seq_prob / len(list)
            for item in list:
                p = random.random()
                if item_prob < 0 or p <= item_prob:
                    list_file.write('%d,%s\n' % (idx, item))
                    valid += 1
        stats = (idx, id, valid)
        id_count.append(stats)
        print(stats)
        idx += 1
    list_file.close()

    # write id_name_list
    with open(id_name_list, 'w') as f:
        for idx, id, valid, in id_count:
            f.write('%d,%s,%d\n' % (idx, id, valid))


def get_parser():
    parser = argparse.ArgumentParser(description='parameters test')
    parser.add_argument('--src', default='.', help='source dir')
    parser.add_argument('--dst', default='', help='dst dir')
    parser.add_argument('--mpc', default='-1000', help='max per class')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    voxceleb2_dir = args.src
    output_dir = args.dst
    max_per_class = float(args.mpc)
    gen_balanced(voxceleb2_dir, output_dir, max_per_class)
