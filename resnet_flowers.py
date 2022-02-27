#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2021/7/28 12:57 
@Author : Gabriel
@File : resnet_flowers.py 
@Project:ImageClassification
@About :
使用resnet实现花卉分类
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2表示只显示错误信息

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import ResNet50

tf.random.set_seed(2345)

# 设置GPU的最大使用量
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)


def load_data(dir):
    '''
    读取数据
    :param dir:str 数据集所在总目录
    :return: list,list 所有数据图片路径，所有图片对应标签
    '''
    data = []  # 存放图片路径
    label = []  # 存放图片类型
    for index, path in enumerate(os.listdir(dir)):
        if os.path.isdir(dir + path):
            for flower_p in os.listdir(dir + path):
                data.append(dir + path + '/' + flower_p)
                label.append(index)  # 添加标签,0:daisy,1:dandelion,2:roses,3:sunflowers,4:tulips

    return data, label


class FlowerClassification:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.batch_size = 32

        self.data_slices()
        self.train()

    def preprocess(self, x, y):
        '''
        数据预处理
        :param x:图片
        :param y:标签
        :return:
        '''

        x = tf.io.read_file(x)
        x = tf.image.decode_jpeg(x)  # 将图片转为矩阵
        x = tf.image.resize(x, [100, 100])
        x = 2 * tf.cast(x, dtype=tf.int32) / 255 - 1
        y = tf.cast(y, dtype=tf.int32)
        y = tf.one_hot(y, depth=5)
        return x, y

    def data_slices(self):
        '''
        数据划分
        :return:
        '''
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(
            self.data, self.label,
            test_size=0.3,
            shuffle=True
        )
        self.data_valid, self.data_test, self.label_valid, self.label_test = train_test_split(
            self.data_test, self.label_test,
            test_size=0.5,
            shuffle=True
        )

        # 训练集
        self.train_db = tf.data.Dataset.from_tensor_slices((self.data_train, self.label_train))
        self.train_db = self.train_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            1234).batch(self.batch_size)

        # 验证集
        self.valid_db = tf.data.Dataset.from_tensor_slices((self.data_valid, self.label_valid))
        self.valid_db = self.valid_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            1234).batch(self.batch_size)

        # 测试集
        self.test_db = tf.data.Dataset.from_tensor_slices((self.data_test, self.label_test))
        self.test_db = self.test_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            self.batch_size)

    def train(self):
        import datetime

        # tensorboard相关配置
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'log/resnet_' + curr_time
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # 使用逆时学习率衰减
        inverse_time_decay = optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-2, decay_rate=0.1, decay_steps=10
        )

        # model = resnet18()
        model = ResNet50(classes=5, weights=None)
        model.build(input_shape=(None, 100, 100, 3))
        model.summary()

        optimizer = optimizers.Adam(1e-3)
        model.compile(optimizer=optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(self.train_db, epochs=10, validation_data=self.valid_db, validation_steps=10,
                  callbacks=[tensorboard])


if __name__ == '__main__':
    path = './flowers/'
    data, label = load_data(path)
    Pred = FlowerClassification(data, label)
