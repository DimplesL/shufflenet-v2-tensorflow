# coding:utf-8
"""
author：Qiu Yurui

"""
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import argparse
import glob
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow


def getweightpath(wdir):
    ckpath = os.path.join(wdir, 'checkpoint')
    fr = open(ckpath, 'rt')
    fline = fr.readline()
    fr.close()
    ckname = fline.split('"')[1]
    return os.path.join(wdir, ckname)


def exportpb_fromckpt(input_checkpoint, output_graph, output_node_names):
    """
    :param input_checkpoint:  ckpt model path
    :param output_graph:   save path of pb model
    :return:
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            gbs = tf.Variable(0, trainable=False)
            input_image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='images')
            label_target = tf.placeholder(tf.int32, shape=[None, ], name='labels')
            logits, end_points = inception_v3.inception_v3(input_image,
                                                           num_classes=2,
                                                           is_training=False,
                                                           dropout_keep_prob=0.0,
                                                           depth_multiplier=0.5)
            # output = tf.identity(logits, name=output_node_names)
            saver = tf.train.Saver()
            saver.restore(sess, input_checkpoint)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=output_node_names.split(','))
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    root_path_model = ''
    root_path_pb = ''
    output_node_names = ''

    checkpoint_path = os.path.join('/Users/qiuyurui/Downloads/model.ckpt-1758393')
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: ', key)
    freeze_graph('/Users/qiuyurui/Downloads/model.ckpt-1758393', 'test.pb')

    if not os.path.exists(root_path_pb):
        os.makedirs(root_path_pb)

    dirs = glob.glob(root_path_model + '/*')
    for dir in dirs:
        if dir.startswith('.'):
            continue
        if not os.path.isdir(dir):
            continue
        number = dir.split('/')[-1].split('_')[-1]
        ckpath = getweightpath(dir)
        pbpath = os.path.join(root_path_pb, '{0}.pb'.format(number))
        exportpb_fromckpt(ckpath, pbpath, output_node_names)
