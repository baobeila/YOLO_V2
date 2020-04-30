#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#-----------------------------------------------------------------------------------
import tensorflow as tf
import argparse
import datetime
import time
import os
import yolo.config as cfg
from tensorflow.python import debug as tfdbg

from yolo.preprocess import Data_preprocess
from six.moves import xrange
from yolo.yolo_v2 import yolo_v2


class Train(object):
    def __init__(self, yolo, data):
        self.yolo = yolo
        self.data = data
        self.num_class = len(cfg.CLASSES)
        self.max_step = cfg.MAX_ITER
        self.saver_iter = cfg.SAVER_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.initial_learn_rate = cfg.LEARN_RATE
        self.output_dir = os.path.join(cfg.DATA_DIR, 'output')

        # weight_file = os.path.join(cfg.DATA_DIR,'pretrained', cfg.WEIGHTS_FILE)
        weight_file = os.path.join(cfg.DATA_DIR,'output', cfg.WEIGHTS_FILE)

        self.variable_to_restore = tf.global_variables()
        #最后一层权重不进行加载,该句是针对预训练权重为voc，与实际检测的类别不同
        # self.variable_to_restore =  tf.contrib.framework.get_variables_to_restore(exclude=['biases_22','weight_22'])

        # vars_save = tf.trainable_variables()
        # g_list = tf.global_variables()
        # bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        # bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # vars_save += bn_moving_vars

        self.saver = tf.train.Saver(self.variable_to_restore)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        if  cfg.LEARN_SET=='exponential_decay':
            self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 2000, 0.96, name='learn_rate')
        else:

            self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 190, 10000, 15500], [self.initial_learn_rate, 5e-3, 1e-2, 1e-3, 1e-4])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if not cfg.CLIP_NORM:
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)
        else:
        # 梯度裁剪
            with tf.control_dependencies(update_ops):

                self.clip_optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

                grads, variables = zip(*self.clip_optimizer.compute_gradients(self.yolo.total_loss))
                grads, global_norm = tf.clip_by_global_norm(grads, 5)
                self.optimizer = self.clip_optimizer.apply_gradients(zip(grads, variables))

        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())

        with tf.control_dependencies([self.optimizer]):

            self.train_op = tf.group(self.average_op)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        self.sess = tf.Session(config=config)
        if cfg.DEBUG:

            self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess,ui_type="readline")
            self.sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

        #下面这句能否去掉？？应该不能
        self.sess.run(tf.global_variables_initializer())
        try :

            self.saver.restore(self.sess, weight_file)
            print('Restore weights from:', weight_file)

        except:
            print("Training from scratch")

        self.writer = tf.summary.FileWriter(self.output_dir+'/train',self.sess.graph)

        self.writer_test =tf.summary.FileWriter(self.output_dir+'/test')
        self.summary_op = tf.summary.merge_all()

    def test_loss(self,labels_test):
        num = 5
        sum_loss = 0
        #可以不取五折平均
        for i in range(num):
            images_t, labels_t = self.data.next_batches_test(labels_test)
            feed_dict_t = {self.yolo.images: images_t, self.yolo.labels: labels_t}
            _summary,loss_t = self.sess.run([self.summary_op,self.yolo.total_loss], feed_dict=feed_dict_t)
            sum_loss += loss_t
        return _summary,sum_loss / num
    def train(self):
        labels_train = self.data.load_labels('train')
        labels_test = self.data.load_labels('test')

        num = 5
        initial_time = time.time()
        #初始化一个预定的测试集loss
        best_loss = 1000
        last_improved = 0
        require_improvement = 2000

        for step in xrange(1, self.max_step + 1):
            images, labels = self.data.next_batches(labels_train)
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels}

            if step % self.summary_iter == 0:

                if step % 50 == 0:
                    #训练50次的倍数后，在测试集上取5个batch做测试，防止刚开始不稳定，测试集上损失为NAN
                    if (step<100):
                        # self.summary_op = tf.summary.merge_all()

                        loss, _ = self.sess.run([ self.yolo.total_loss, self.train_op],feed_dict=feed_dict)
                        log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f},  Remain: {}').format(
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss,
                             self.remain(step, initial_time))
                        print(log_str)
                    else:
                        #50 100 150
                        test_summary,average_test_loss = self.test_loss(labels_test)
                        # tf.summary.scalar('test_loss', average_test_loss)
                        self.writer_test.add_summary(test_summary,step)
                        #2000次损失不下降，早停，防止过拟合
                        if (average_test_loss< best_loss ):
                            best_loss = average_test_loss
                            last_improved = step
                        if step - last_improved > require_improvement:
                            print("Early-stop",step,"bestloss",best_loss)
                            break

                        #50 100 训练
                        summary_, loss, _ = self.sess.run([self.summary_op, self.yolo.total_loss, self.train_op], feed_dict = feed_dict)

                        self.writer.add_summary(summary_, step)

                        log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f}, test_Loss: {:.4f}, Remain: {}').format(
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss, average_test_loss, self.remain(step, initial_time))
                        print(log_str)


                else:
                    # self.summary_op = tf.summary.merge_all()
                    #10  20 30 40 60 70 80 90
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)

                    self.writer.add_summary(summary_, step)

            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)

            if step % self.saver_iter == 0:

                #必须重新定义saver 保存所有的变量
                #设置保存模型
                vars_save = tf.trainable_variables()
                g_list = tf.global_variables()
                bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
                bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
                vars_save+=bn_moving_vars
                tf.train.Saver(var_list=vars_save).save(self.sess, self.output_dir + '/yolo_v2.ckpt', global_step = step)

    def remain(self, i, start):
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt-4000', type = str)  # darknet-19.ckpt
    # parser.add_argument('--weights', default = '', type = str)  # darknet-19.ckpt
    #voc的预训练权重
    # parser.add_argument('--weights', default = 'yolo_weights.ckpt', type = str)
    parser.add_argument('--gpu', default = "0", type = str)  # which gpu to be selected
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS_FILE = args.weights
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo_obj = yolo_v2()


    pre_data = Data_preprocess()

    train = Train(yolo_obj, pre_data)

    print('start training ...')
    train.train()
    print('successful training.')


if __name__ == '__main__':
    main()
