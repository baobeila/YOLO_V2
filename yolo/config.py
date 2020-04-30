#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari

DATA_DIR = 'data'
DATA_SET = 'data_set/defect_data'
WEIGHTS_FILE = ''

CLASSES = ['aoxian', 'cahuashang', 'hanzha', 'huashang', 'liewen', 'shuizhayin',
           'zhapi']


DEBUG=False
# DEBUG=True
# 注意宽和高的排布，前五个为宽，后五个为高
ANCHOR = [2.5489, 0.65 , 1.972899, 7.336494, 0.975, 4.09318, 1.22776, 1.715542,10.532453, 2.294075]

GPU = '0,1'

IMAGE_SIZE = 416    #The size of the input images

LEARN_SET = 'exponential_decay'
LEARN_RATE = 0.0001     #The learn_rate of training
MAX_ITER = 1000    #The max step
SUMMARY_ITER = 10    #Every 'summary_iter' step output a summary
SAVER_ITER = 1000    #Every 'saver_iter' step save a weights
CLIP_NORM=False     #loss波动较大时使用
BOX_PRE_CELL = 5    #The number of BoundingBoxs predicted by each grid
CELL_SIZE = 13      #The size of the last layer  #(batch_size, 13, 13, ?)
BATCH_SIZE = 16     #The batch size of each training
ALPHA = 0.1
LEARN_ANCHOR_ITER=0
THRESHOLD = 0.2    #The threshold of the probability of the classes,根据实际情况调整
# [[ 2.54893067  4.09318182]
#  [ 0.65        1.2277655 ]
#  [ 1.97289976  1.71554253]
#  [ 7.33649436 10.53245375]
#  [ 0.975       2.29407529]]