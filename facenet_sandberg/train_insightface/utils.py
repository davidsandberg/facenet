import argparse
import os
import pickle
import time
from enum import Enum, auto
from os.path import exists, join
from typing import List, Tuple, Union, cast

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorlayer as tl

from config import Config
from tensorflow_extractor import TensorflowExtractor
from verification import extract_list_feature, verification

FaceVector = List[float]
Match = Tuple[str, int, int]
Mismatch = Tuple[str, int, str, int]
Pair = Union[Match, Mismatch]
Path = Tuple[str, str]
Label = bool
Image = np.ndarray
ImagePairs = List[Tuple[Image, Image]]


def load_lfw(config: Config) -> Tuple[ImagePairs, ImagePairs]:
    print('Loading lfw data:')
    pairs, _, num_matches_mismatches = _read_pairs(config.get('lfw').pairs)
    pair_paths, labels = _get_paths_and_labels(
        config.get('lfw').image_dir, pairs)
    pos_img, neg_img = split(
        config.get('lfw').image_dir, pair_paths, labels, image_size)

    # crop image
    pos_img = crop_image_list(pos_img, image_size)
    neg_img = crop_image_list(neg_img, image_size)
    return pos_img, neg_img


def ver_test(pos_list: ImagePairs, neg_list: ImagePairs,
             extractor: TensorflowExtractor):
    pos_feat = extract_list_feature(
        extractor,
        pos_list,
        len(pos_list),
        extractor.batch_size)
    neg_feat = extract_list_feature(
        extractor,
        neg_list,
        len(neg_list),
        extractor.batch_size)
    _acc, _std, _threshold, _pos, _neg, _accu_list = verification(
        pos_feat, neg_feat, 'cosine')
    return _accu_list, _acc, _std


def crop_image_list(
        img_list: List[Tuple[Image, Image]], imsize: Tuple[int, int]):
    out_list = []
    h, w, c = img_list[0][0].shape
    x1 = (w - imsize[0]) // 2
    y1 = (h - imsize[1]) // 2
    for pair in img_list:
        img1 = pair[0]
        img2 = pair[1]
        img1 = img1[y1:(y1 + imsize[1]), x1:(x1 + imsize[0]), :]
        img1 = (np.float32(img1) - 127.5) / 128
        img2 = img2[y1:(y1 + imsize[1]), x1:(x1 + imsize[0]), :]
        img2 = (np.float32(img2) - 127.5) / 128
        out_list.append([img1, img2])
    return out_list


def _read_pairs(pairs_filename: str) -> Tuple[List[Pair], int, int]:
    pairs = []
    with open(pairs_filename, 'r') as pair_file:
        num_sets, num_matches_mismatches = [int(i)
                                            for i in next(pair_file).split()]
        for line_num, line in enumerate(pair_file):
            pair = cast(Pair, tuple([int(i) if i.isdigit() else i
                                     for i in line.strip().split()]))
            pairs.append(pair)
    return pairs, num_sets, num_matches_mismatches


def _get_paths_and_labels(image_dir: str,
                          pairs: List[Pair]) -> Tuple[List[Path], List[Label]]:
    paths = []
    labels = []
    for pair in pairs:
        _add_extension = (lambda rel_image_path, image_dir:
                          f'{rel_image_path}.jpg'
                          if exists(join(image_dir, f'{rel_image_path}.jpg'))
                          else f'{rel_image_path}.png')
        if len(pair) == 3:
            person, image_num_0, image_num_1 = cast(Match, pair)
            rel_image_path_no_ext = join(person,
                                         f'{person}_{image_num_0:04d}')
            rel_image_path_0 = _add_extension(rel_image_path_no_ext, image_dir)
            rel_image_path_no_ext = join(person,
                                         f'{person}_{image_num_1:04d}')
            rel_image_path_1 = _add_extension(rel_image_path_no_ext, image_dir)
            is_same_person = True
        elif len(pair) == 4:
            person_0, image_num_0, person_1, image_num_1 = cast(Mismatch, pair)
            rel_image_path_no_ext = join(person_0,
                                         f'{person_0}_{image_num_0:04d}')
            rel_image_path_0 = _add_extension(rel_image_path_no_ext, image_dir)
            rel_image_path_no_ext = join(person_1,
                                         f'{person_1}_{image_num_1:04d}')
            rel_image_path_1 = _add_extension(rel_image_path_no_ext, image_dir)
            is_same_person = False
        if (exists(join(image_dir, rel_image_path_0))
                and
                exists(join(image_dir, rel_image_path_1))):
            paths.append((rel_image_path_0, rel_image_path_1))
            labels.append(is_same_person)
        else:
            err = f'{rel_image_path_no_ext} with .jpg or .png extensions'
            raise FileNotFoundError(err)
    return paths, labels


def split(base_path: str,
          paths: List[Path],
          labels: List[Label],
          image_size: Tuple[int,
                            int]) -> Tuple[ImagePairs, ImagePairs]:
    pos = []
    neg = []
    for index in range(len(paths)):
        img1 = cv2.imread(os.path.join(base_path, paths[index][0]))
        img1 = cv2.resize(img1, (image_size[0], image_size[1]))
        img2 = cv2.imread(os.path.join(base_path, paths[index][1]))
        img2 = cv2.resize(img2, (image_size[0], image_size[1]))
        if labels[index]:
            pos.append((img1, img2))
        else:
            neg.append((img1, img2))
    return pos, neg


def load_image_list(pair_list):
    img_list = []
    for pair in pair_list:
        # skip invalid pairs
        if not os.path.exists(pair[0]) or not os.path.exists(pair[1]):
            continue
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        img_list.append([img1, img2, pair[0], pair[1]])
    return img_list


def load_ytf_pairs(path, prefix):
    pos_list_ = []
    neg_list_ = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            flag, a, b = line.split(',')
            flag = int(flag)
            a = os.path.join(prefix, a)
            b = os.path.join(prefix, b)
            if flag == 1:
                pos_list_.append([a, b])
            else:
                neg_list_.append([a, b])

    pos_img = load_image_list(pos_list_)
    neg_img = load_image_list(neg_list_)
    return pos_img, neg_img
