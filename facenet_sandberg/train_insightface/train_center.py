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
from facenet_sandberg.models.L_Resnet_E_IR_fix_issue9 import get_resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import KFold
from tensorflow.core.protobuf import config_pb2

from config import Config
from dataset_all import build_dataset
from EMA import EMA
from face_losses import arcface_loss, softmax_loss
from mt_loader import MultiThreadLoader
from tensorflow_extractor import TensorflowExtractor
from utils import *
from verification import extract_list_feature, verification

FaceVector = List[float]
Match = Tuple[str, int, int]
Mismatch = Tuple[str, int, str, int]
Pair = Union[Match, Mismatch]
Path = Tuple[str, str]
Label = bool
Image = np.ndarray
ImagePairs = List[Tuple[Image, Image]]


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--config', default='config.ini', help='config file')
    parser.add_argument(
        '--net_depth',
        default=50,
        help='resnet depth, default is 50')
    parser.add_argument(
        '--epoch',
        default=100000,
        help='epoch to train the network')
    parser.add_argument(
        '--batch_size',
        default=64,
        help='batch size to train network')
    parser.add_argument(
        '--lr_steps',
        default=[
            40000,
            60000,
            80000],
        help='learning rate to train network')
    parser.add_argument(
        '--momentum',
        default=0.9,
        help='learning alg momentum')
    parser.add_argument(
        '--weight_deacy',
        default=1e-4,
        help='learning alg momentum')

    parser.add_argument(
        '--image_size',
        default=[
            112,
            112],
        help='image size height, width')
    parser.add_argument('--num_output', default=85164, help='the image size')

    parser.add_argument(
        '--summary_path',
        default='./output/summary',
        help='the summary file save path')
    parser.add_argument(
        '--ckpt_path',
        default='./output/ckpt',
        help='the ckpt file save path')
    parser.add_argument(
        '--log_file_path',
        default='./output/logs',
        help='the ckpt file save path')
    parser.add_argument(
        '--saver_maxkeep',
        default=100,
        help='tf.train.Saver max keep ckpt files')
    parser.add_argument(
        '--log_device_mapping',
        default=True,
        help='show device placement log')
    parser.add_argument(
        '--summary_interval',
        default=300,
        help='interval to save summary')
    parser.add_argument(
        '--ckpt_interval',
        default=20000,
        help='intervals to save ckpt file')
    parser.add_argument(
        '--validate_interval',
        default=2000,
        help='intervals to save ckpt file')
    parser.add_argument(
        '--show_info_interval',
        default=20,
        help='intervals to save ckpt file')
    # triplet
    parser.add_argument(
        '--model_path',
        default=None,
        help='baseline model ckpt')
    parser.add_argument(
        '--centre_weight',
        default=0.3,
        help='centre loss weight')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = get_parser()
    model_path = args.model_path
    best_lfw = 0
    count = 0
    if model_path:
        _segs = model_path.split('_')
        count = int(_segs[1])
        best_lfw = float(_segs[3])
    print('best lfw accuracy is %.5f' % best_lfw)
    print('iteration:%d' % count)

    # 1. define global parameters
    image_size = (args.image_size[1], args.image_size[0])
    global_step = tf.Variable(
        name='global_step',
        initial_value=0,
        trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    images = tf.placeholder(
        name='img_inputs',
        shape=[
            None,
            args.image_size[0],
            args.image_size[1],
            3],
        dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    # trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    # load config
    config = Config(args.config)

    # 2 prepare train datasets and test datasets by using tensorflow dataset api
    # 2.1 train datasets
    # the image is substracted 127.5 and multiplied 1/128.
    # random flip left right
    args.ckpt_path = './output/vgg-ms1m'
    dataset_list = []
    dataset_list.append(('vgg', -1))
    dataset_list.append(('ms1m', -1))
    # dataset_list.append(('WebFace', -1))
    dataset = build_dataset(config, dataset_list, balance=False)
    db = MultiThreadLoader(dataset, args.batch_size, 1)
    args.num_output = db.numOfClass()
    batch_per_epoch = db.size() / args.batch_size

    # 2.2 prepare validate datasets
    # lfw
    if config.get('lfw').enable:
        pos_img, neg_img = load_lfw(config)

    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    print('Buiding net structure')
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(
        images,
        args.net_depth,
        type='ir',
        w_init=w_init_method,
        trainable=True,
        keep_rate=dropout_rate)

    # 3.2 get arcface loss
    logit, inner_sim = arcface_loss(
        embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)
    # test net  because of batch normal layer
    tl.layers.set_name_reuse(True)
    test_net = get_resnet(
        images,
        args.net_depth,
        type='ir',
        w_init=w_init_method,
        trainable=False,
        reuse=True,
        keep_rate=dropout_rate)
    embedding_tensor = test_net.outputs

    # 3.3 define the cross entropy
    inference_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=labels))
    wd_loss = 0
    for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for W in tl.layers.get_variables_with_name(
            'resnet_v1_50/E_DenseLayer/W', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
    for weights in tl.layers.get_variables_with_name(
            'embedding_weights', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
    for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
    # for beta in tl.layers.get_variables_with_name('beta', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(beta)
    for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
    # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
    #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(bias)

    # 3.5 total losses
    inner_loss = (1 - inner_sim / args.batch_size) * 64 * args.centre_weight
    total_loss = inference_loss + wd_loss + inner_loss

    # 3.6 define the learning rate schedule
    p = int(512.0 / args.batch_size)
    lr_steps = [p * val for val in args.lr_steps]
    # print(lr_steps)
    lr = tf.train.piecewise_constant(
        global_step, boundaries=lr_steps, values=[
            0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')

    # 3.7 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

    # 3.8 get train op
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = opt.minimize(total_loss, global_step=global_step)

    # 3.9 define the inference accuracy used during validate or test
    pred = tf.nn.softmax(logit)
    train_acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(
                    pred,
                    axis=1),
                labels),
            dtype=tf.float32))

    # 3.10 define sess
    # sess = tf.Session()
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_mapping)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    # 3.11 summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []

    # # 3.11.1 add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(
                tf.summary.histogram(
                    var.op.name +
                    '/gradients',
                    grad))

    # 3.11.2 add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    # 3.11.3 add loss summary
    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))

    # 3.11.4 add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    summary_op = tf.summary.merge(summaries)

    # 3.12 saver
    saver = tf.train.Saver(max_to_keep=args.saver_maxkeep)

    # 3.13 init all variables
    sess.run(tf.global_variables_initializer())

    # restore weights
    learn_vars = tf.trainable_variables()
    model_vars = []
    for var in learn_vars:
        if var.name.find('_loss') < 0:
            model_vars.append(var)
    model_vars = learn_vars
    # latest_checkpoint = get_latest_checkpoint(model_path)
    latest_checkpoint = model_path
    # latest_checkpoint = None
    if latest_checkpoint is not None:
        restore = slim.assign_from_checkpoint_fn(
            latest_checkpoint, var_list=model_vars, ignore_missing_vars=True)
        # restore(sess)
        saver.restore(sess, latest_checkpoint)

    # 4 begin iteration
    ema_iloss = EMA()
    ema_tloss = EMA()
    ema_acc = EMA()
    print('\n\nTraining started ...')
    for i in range(args.epoch):
        for batch in range(batch_per_epoch):
            try:
                # get batch
                images_train, labels_train = db.getBatch()
                # print(images_train)
                images_train = (np.float32(images_train) - 127.5) / 128
                # images_train = np.float32(images_train)
                # print(images_train)
                feed_dict = {
                    images: images_train,
                    labels: labels_train,
                    dropout_rate: 0.4}
                feed_dict.update(net.all_drop)
                start = time.time()

                _, total_loss_val, inference_loss_val, wd_loss_val, inner_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, wd_loss, inner_loss, inc_op, train_acc],
                             feed_dict=feed_dict)

                end = time.time()
                pre_sec = args.batch_size / (end - start)
                inference_loss_val = ema_iloss(inference_loss_val)
                total_loss_val = ema_tloss(total_loss_val)
                acc_val = ema_acc(acc_val)
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print(
                        '%2d, %d/%d, loss:%.2f , iloss:%.2f, wloss:%.2f, inner:%.2f, acc:%.4f, speed:%.1f' %
                        (i,
                         batch,
                         batch_per_epoch,
                         total_loss_val,
                         inference_loss_val,
                         wd_loss_val,
                         inner_loss_val,
                         acc_val,
                         pre_sec))

                # save summary
                '''
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    feed_dict.update(net.all_drop)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)
                '''

                # lfw validate
                is_model_good = False
                model_lfw = 0
                if config.get(
                        'lfw').enable and count > 0 and count % args.validate_interval == 0:
                    feed_dict_test = {dropout_rate: 1.0}
                    feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    extractor = TensorflowExtractor(
                        sess, embedding_tensor, args.batch_size, feed_dict, images)
                    results, precision, _std = ver_test(
                        pos_img, neg_img, extractor)
                    print(
                        '------------------------------------------------------------')
                    print(
                        'Precision on %s : %1.5f+-%1.5f' %
                        ('lfw', precision, _std))
                    model_lfw = precision
                    if precision > best_lfw:
                        best_lfw = precision
                        if precision > 0.99:
                            is_model_good = True
                    print('best lfw accuracy is %.5f' % best_lfw)
                    print('\n')

                # save ckpt files
                if is_model_good or count % args.ckpt_interval == 0:
                    filename = 'iter_%d_lfw_%.5f' % (
                        count, model_lfw) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)

                count += 1
            except Exception as e:
                print(e)
