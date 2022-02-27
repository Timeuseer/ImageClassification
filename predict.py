#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2021/8/12 16:09 
@Author : Gabriel
@File : predict.py 
@Project:ImageClassification
@About : 使用模型预测花的种类
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2表示只显示错误信息

import tensorflow as tf
import numpy as np
import wrapt
from line_profiler import LineProfiler

lp = LineProfiler()
tf.compat.v1.enable_eager_execution()


def lp_wrapper():
    '''
    显示每一步的调用时间
    :return:
    '''

    @wrapt.decorator
    def wrapper(func, instance, args, kwargs):
        global lp
        lp_wrapper = lp(func)
        res = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return res

    return wrapper


def preprocess(path):
    '''
    对图片进行预处理
    :param path:
    :return:
    '''
    imgs = []
    for x in tf.io.gfile.glob(path + '*.jpg'):
        img = tf.io.read_file(x)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [100, 100])
        # img = tf.expand_dims(img, axis=0)  # [100,100,3]->[1,100,100,3]
        imgs.append(img)
    imgs = tf.convert_to_tensor(np.array(imgs))
    return imgs


# @lp_wrapper()
def main(path=''):
    # 导入模型
    imported = tf.saved_model.load('./DenseNetModel/')

    model = imported.signatures['serving_default']
    img = preprocess(path)

    preds = model(img)
    for key, value in preds.items():
        pred = tf.argmax(preds[key], axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        for x in pred:
            print(label[int(x)])
        # print(pred)


if __name__ == '__main__':
    label = {
        0: 'daisy',
        1: 'dandelion',
        2: 'roses',
        3: 'sunflowers',
        4: 'tulips'
    }
    main('./test/')
