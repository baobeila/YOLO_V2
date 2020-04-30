#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#-----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import argparse
import colorsys
import cv2
import os
import time
import yolo.config as cfg
from yolo.yolo_v2 import yolo_v2



class Detector(object):
    def __init__(self, yolo, weights_file):
        self.yolo = yolo
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.threshold = cfg.THRESHOLD
        self.anchor = cfg.ANCHOR



        #默认会加载所有变量
        self.saver = tf.train.Saver()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))

        self.sess = tf.Session(config=config)
        # if tf.train.latest_checkpoint('data/output') is not None:
        #     self.saver.restore(self.sess, tf.train.latest_checkpoint('data/output'))
        #     print('Restore weights from: ' + 'data/output')
        # else:
        #     assert 'can not find checkpoint folder path!'
        # self.sess.run(tf.global_variables_initializer())
        print('Restore weights from: ' + weights_file)
        self.saver.restore(self.sess, weights_file)

    def detect(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        output = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: image})

        results = self.calc_output(output)

        for i in range(len(results)):
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)

        return results


    def calc_output(self, output):
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])#13 13 5 12
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])    #boxes coordinate
        boxes = self.get_boxes(boxes) * self.image_size

        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])    #the confidence of the each anchor boxes
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))

        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])    #classes
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))#softmax

        probs = classes * confidence#13 13 5 7

        filter_probs = np.array(probs >= self.threshold, dtype = 'bool')#13 13 5 7
        filter_index = np.nonzero(filter_probs)#返回的是元组，有四个元素，每个元素为数组，第一个元素代表第0维
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]#对应大于阈值所在的box
        probs_filter = probs[filter_probs]#找出大于阈值所对应的可能性
        #大于阈值所对应的类别
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]

        sort_num = np.array(np.argsort(probs_filter))[::-1]#按照置信度排序  类别 2 0 1
        box_filter = box_filter[sort_num]#对应的框
        probs_filter = probs_filter[sort_num]#排序过后对应的可能性
        classes_num = classes_num[sort_num]
        #完成nms功能
        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0

        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append([self.classes[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i]])

        return results

    def get_boxes(self, boxes):
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 5]) / self.cell_size,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 5]) / self.cell_size])
        #box1中出现了大于1的值，表现很差
        return np.transpose(boxes1, (1, 2, 3, 0))


    def calc_iou(self, box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors


    def draw(self, image, result):
        image_h, image_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))


    def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imwrite("./photo_{}".format(os.path.split(imagename)[1]), image)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)


    def video_detect(self, cap):
        while(1):
            ret, image = cap.read()
            if not ret:
                print('Cannot capture images from device')
                break

            result = self.detect(image)
            self.draw(image, result)
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    start0 = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt-4000', type = str)    # darknet-19.ckpt
    parser.add_argument('--weight_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '0', type = str)    # which gpu to be selected
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    # configure gpu
    weights_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    # yolo_obj = yolo_v2(False)    # 'False' mean 'test'
    yolo_obj = yolo_v2(True)    # 单张图片预测，设置True效果好


    detector = Detector(yolo_obj, weights_file)
    elapsed = (time.clock() - start0)
    print("Time used:", elapsed)
    #detect the video
    #cap = cv2.VideoCapture('asd.mp4')
    #cap = cv2.VideoCapture(0)
    #detector.video_detect(cap)

    #detect the image
    start1 = time.clock()
    file_path = os.getcwd() + r'/data/data_set/defect_data/train_image'
    for i in os.listdir(file_path):
        imagename =  os.path.join(file_path,i)
        detector.image_detect(imagename)
    elapsed1 = (time.clock() - start1)
    print("Time used:", elapsed1)



if __name__ == '__main__':
    main()
