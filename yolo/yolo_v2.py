#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
# -----------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import yolo.config as cfg



class yolo_v2(object):
    def __init__(self, isTraining=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.isTraining = isTraining
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.anchor = cfg.ANCHOR
        self.alpha = cfg.ALPHA

        self.class_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.coordinate_scale = 1.0
        # 前12800次权重系数（具体数值查看原论文）
        self.prior_scale = 1.0

        self.offset = np.transpose(
            np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                       [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))  # 数据格式转换
        self.offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32),
                                 [1, self.cell_size, self.cell_size, self.box_per_cell])
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))

        # self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.images=tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_networks(self.images,self.isTraining)
        self.batch_count = 0
        if self.isTraining:
            self.labels = tf.compat.v1.placeholder(tf.float32,
                                         [None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5],
                                         name='labels')


            tf.add_to_collection('losses', self.loss_layer(self.logits, self.labels))

            self.total_loss = tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('total_loss', self.total_loss)

    def build_networks(self, inputs,isTraining):
        net = self.conv_layer(inputs, [3, 3, 3, 32], isTraining,name='0_conv')
        net = self.pooling_layer(net, name='1_pool')

        net = self.conv_layer(net, [3, 3, 32, 64], isTraining,name='2_conv')
        net = self.pooling_layer(net, name='3_pool')

        net = self.conv_layer(net, [3, 3, 64, 128], isTraining,name='4_conv')
        net = self.conv_layer(net, [1, 1, 128, 64], isTraining,name='5_conv')
        net = self.conv_layer(net, [3, 3, 64, 128],isTraining, name='6_conv')
        net = self.pooling_layer(net, name='7_pool')

        net = self.conv_layer(net, [3, 3, 128, 256], isTraining,name='8_conv')
        net = self.conv_layer(net, [1, 1, 256, 128], isTraining,name='9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], isTraining,name='10_conv')
        net = self.pooling_layer(net, name='11_pool')

        net = self.conv_layer(net, [3, 3, 256, 512], isTraining,name='12_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], isTraining,name='13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], isTraining,name='14_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], isTraining,name='15_conv')
        net16 = self.conv_layer(net, [3, 3, 256, 512], isTraining,name='16_conv')
        net = self.pooling_layer(net16, name='17_pool')

        net = self.conv_layer(net, [3, 3, 512, 1024],isTraining, name='18_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512],isTraining, name='19_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], isTraining,name='20_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512],isTraining, name='21_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024],isTraining, name='22_conv')

        net = self.conv_layer(net, [3, 3, 1024, 1024],isTraining, name='23_conv')
        net24 = self.conv_layer(net, [3, 3, 1024, 1024],isTraining, name='24_conv')

        net = self.conv_layer(net16, [1, 1, 512, 64], isTraining,name='26_conv')
        net = self.reorg(net)

        net = tf.concat([net, net24], 3)

        net = self.conv_layer(net, [3, 3, int(net.get_shape()[3]), 1024], isTraining,name='29_conv')
        net = self.conv_layer(net, [1, 1, 1024, self.box_per_cell * (self.num_class + 5)], None,batch_norm=False,
                              name='30_conv')
        # 最后一层没有经过激活
        return net

    def conv_layer(self, inputs, shape, isTraining,batch_norm=True, name='0_conv',init_method='truncated_normal'):
        #激活函数使用 sigmoid 和 tanh--'xavier'
        #relu he_initialization
        if init_method=='xavier':
            weight = tf.Variable(tf.contrib.layers.xavier_initializer()((
                [shape[0],shape[1],shape[2], shape[3]])), name="weight")
            biases = tf.Variable(tf.contrib.layers.xavier_initializer()(([ shape[3]])), name="biases")
        elif init_method=='truncated_normal':
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            biases = tf.Variable(tf.constant(0.1, shape=[shape[3]]), name='biases')
        elif init_method=='he_initialization':
            weight = tf.Variable(tf.contrib.layers.variance_scaling_initializer()((
                [shape[0], shape[1], shape[2], shape[3]])), name="weight")
            biases = tf.Variable(tf.contrib.layers.variance_scaling_initializer()(([shape[3]])), name="biases")

        # add_to_collection()函数将新生成变量的L2正则化损失加入集合losses
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0000005)(weight))
        # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0000005)(biases))
        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
        conv_bis = tf.add(conv, biases)
        if batch_norm:
            if isTraining is True:
                # 训练模式 使用指数加权函数不断更新均值和方差,将不是训练变量的均值和方差放入一个集合
                conv_bn=tf.layers.batch_normalization(inputs=conv_bis, momentum=0.9,
                                                     # updates_collections=None,
                                                     training=True,
                                                     # zero_debias_moving_mean=True,
                                                     # variables_collections=[
                                                     #     "batch_norm_non_trainable_variables_collection"]
                                                     )
            else:
                # 测试模式 不更新均值和方差，直接使用
                conv_bn= tf.layers.batch_normalization(inputs=conv_bis, momentum=0.9,
                                                      # updates_collections=None,
                                                    training=False,
                                                    # zero_debias_moving_mean=True,
                                                    #   variables_collections=[
                                                    #       "batch_norm_non_trainable_variables_collection"]
                                                      )

            conv = tf.maximum(self.alpha * conv_bn, conv_bn)
            return conv

        else:
            return conv_bis

    def pooling_layer(self, inputs, name='1_pool'):
        pool = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool

    def reorg(self, inputs):

        return tf.space_to_depth(inputs, block_size=2)

    def loss_layer(self, predict, label):
        # 解码输出
        predict = tf.reshape(predict,
                             [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5])
        # 中心坐标相对于cell左上角的x坐标和y坐标的偏置，后面采用sigmod是为了将偏置控制在0和1之间
        box_coordinate = tf.reshape(predict[:, :, :, :, :4],
                                    [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        # 存储的是置信度，采用sigmod控制在0和1之间
        box_confidence = tf.reshape(predict[:, :, :, :, 4],
                                    [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 1])
        box_classes = tf.reshape(predict[:, :, :, :, 5:],
                                 [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        #  batch 13 13 5 4
        # 下面三项得到预测输出
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_classes = tf.nn.softmax(box_classes)

        response = tf.reshape(label[:, :, :, :, 0],
                              [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        # ground_truth batch 13 13 5 4
        boxes = tf.reshape(label[:, :, :, :, 1:5],
                           [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        classes = tf.reshape(label[:, :, :, :, 5:],
                             [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])
        box_coor_trans = self.decode_box(box_coordinate)
        iou = self.calc_iou(box_coor_trans, boxes)

        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
        # 直接选出了五个annchor中哪一个anchor 是最合适的 mask
        confs = tf.expand_dims(best_box * response, axis=4,name='confs_origin')
        #下面一句经过修改，找到最合适anchor的IOU
        confs_iou = tf.expand_dims(best_box * iou * response, axis=4,name='confs_iou')

        conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
        cooid = self.coordinate_scale * confs
        proid = self.class_scale * confs

        coo_loss = cooid * tf.square(box_coor_trans - boxes)
        con_loss = conid * tf.square(box_confidence - confs_iou)
        pro_loss = proid * tf.square(box_classes - classes)
        if (self.batch_count < cfg.LEARN_ANCHOR_ITER):

            # 记录anchor框，该部分损失是为了学习anchor形状
            prior_box = tf.stack([(0.5 + self.offset) / self.cell_size,
                                  (0.5 + tf.transpose(self.offset, (
                                      0, 2, 1, 3))) / self.cell_size,
                                  tf.to_float(tf.sqrt(tf.ones_like(self.offset) * np.reshape(self.anchor[:5], [1, 1, 1,
                                                                                                               5]) / self.cell_size)),
                                  tf.to_float(tf.sqrt(tf.ones_like(self.offset) * np.reshape(self.anchor[5:], [1, 1, 1,
                                                                                                               5]) / self.cell_size))])
            prior_box_coor_trans = tf.transpose(prior_box, (1, 2, 3, 4, 0))
            prior_loss = self.prior_scale * tf.square(prior_box_coor_trans - boxes)
            loss = tf.concat([coo_loss, con_loss, pro_loss, prior_loss], axis=4)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]), name='loss')
            self.batch_count += 1
        else:
            loss = tf.concat([coo_loss, con_loss, pro_loss], axis=4)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]), name='loss')

        return loss

    def calc_iou(self, boxes1, boxes2):
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        # tf.expand_dims最后一个维度扩维，tf.concat最后一个维度拼接，也可实现
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)

    def decode_box(self, box_coordinate):
        #1e-8可防止LOSS为NAN
        boxes1 = tf.stack([(tf.sigmoid(box_coordinate[:, :, :, :, 0]) + self.offset) / self.cell_size,#沿着w方向增
                           (tf.sigmoid(box_coordinate[:, :, :, :, 1]) + tf.transpose(self.offset, (
                               0, 2, 1, 3))) / self.cell_size,
                           tf.sqrt(1e-8+tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5],
                                                                                      [1, 1, 1, 5]) / self.cell_size),
                           tf.sqrt(1e-8+tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:],
                                                                                      [1, 1, 1, 5]) / self.cell_size)])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))

        return box_coor_trans


if __name__ == '__main__':
    x = np.random.rand(1, 416, 416, 3)
    yolo = yolo_v2()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 必须先restore模型才能打印shape;导入模型时，上面每层网络的name不能修改，否则找不到
        saver.restore(sess, "../data/output/yolo_v2.ckpt-4000")
        feed_dict = {yolo.images: x}
        print(sess.run(yolo.logits, feed_dict=feed_dict).shape)  # (1, 13, 13, 60) 5*(5+7)
