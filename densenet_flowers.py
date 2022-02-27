#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2021/8/9 17:52 
@Author : Gabriel
@File : densenet_flowers.py 
@Project:ImageClassification
@About : 使用densenet实现花卉分类
'''

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2表示只显示错误信息
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, losses
from densenet import densenet
from tensorflow.keras.applications.densenet import DenseNet121
from sklearn.model_selection import train_test_split
from tensorflow import keras

# tf.config.set_soft_device_placement = False
tf.random.set_seed(2345)
tf.compat.v1.enable_eager_execution()


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
    label = []  # 存放图片类型,0:daisy,1:dandelion,2:roses,3:sunflowers,4:tulips
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
        self.batch_size = 128

        self.data_slices()
        self.train()  # 训练
        # self.predict()  # 预测

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
        x = 2 * tf.cast(x, dtype=tf.float32) / 255 - 1
        y = tf.cast(y, dtype=tf.int32)
        y = tf.one_hot(y, depth=5)
        return x, y

    def data_slices(self):
        '''
        数据划分
        :return:
        '''
        self.data_train, self.data_valid, self.label_train, self.label_valid = train_test_split(
            self.data, self.label,
            test_size=0.1,
            shuffle=True
        )
        self.data_valid, self.data_test, self.label_valid, self.label_test = train_test_split(
            self.data_valid, self.label_valid,
            test_size=0.5,
            shuffle=True
        )

        # 训练集
        self.train_db = tf.data.Dataset.from_tensor_slices((self.data_train, self.label_train))
        self.train_db = self.train_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            1234).batch(self.batch_size)
        self.train_db.prefetch(buffer_size=5)

        # 验证集
        self.valid_db = tf.data.Dataset.from_tensor_slices((self.data_valid, self.label_valid)).repeat()
        self.valid_db = self.valid_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            1234).batch(self.batch_size)
        self.valid_db.prefetch(buffer_size=5)

        # 测试集
        self.test_db = tf.data.Dataset.from_tensor_slices((self.data_test, self.label_test))
        self.test_db = self.test_db.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            self.batch_size)

    def train(self):
        import datetime

        # tensorboard相关配置
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'log/densenet_' + curr_time
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # 学习率衰减
        # 1、分段常数衰减
        boundaries = [10, 50, 70]
        values = [0.005, 0.003, 0.001, 0.0005]
        piece_wise_constant_decay = optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=values, name=None
        )
        # 2、逆时衰减 lr = lr0 * (1/(1+decay_rate*(t/decay_step)))
        inverse_time_decay = optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-3, decay_rate=0.01, decay_steps=self.batch_size*100
        )
        # 3、指数衰减  lr = lr0 *(decay_rate^(t/decay_step))
        exponential_decay = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_rate=0.96, decay_steps=1
        )
        # 4、余弦衰减
        cosine_decay = keras.experimental.CosineDecay(
            initial_learning_rate=1e-3, decay_steps=1
        )

        model = densenet(num_classes=5)
        model.build(input_shape=(None, 300, 300, 3))
        model.summary()
        # 绘制网络流程图
        keras.utils.plot_model(model, "./NetImage/DenseNet.png",show_shapes=True,show_layer_names=True)
        # optimizer = optimizers.Adam(1e-3)
        optimizer = optimizers.Adam(inverse_time_decay)  # 自衰减学习率
        model.compile(optimizer=optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        model.fit(self.train_db, epochs=100, validation_data=self.valid_db, validation_steps=10,
                  callbacks=[tensorboard])
        # model.fit(self.train_db, epochs=100, validation_data=self.valid_db, validation_steps=10)

        # 保存模型
        # tf.saved_model.save(model, './DenseNetModel/')    #方法1
        model.save_weights('./DenseNetWeights_V1/')

        pred = model.predict(self.test_db)
        pred = tf.argmax(pred, axis=1)
        acc = tf.keras.metrics.binary_accuracy(y_pred=pred, y_true=self.label_test)
        print(float(acc))

    def predict(self):
        model = densenet()
        model.build(input_shape=(None, 100, 100, 3))
        model.load_weights('./DenseNetWeights_V1/')

        pred = model.predict(self.test_db)
        pred = tf.argmax(pred, axis=1)
        pred = pred.numpy().tolist()

        # count_true = 0
        # count_all = len(pred)
        # for index, data in enumerate(pred):
        #     if self.label_test[index] == data:
        #         count_true += 1
        #
        # accuracy = count_true/count_all
        # print(accuracy)

        m = tf.keras.metrics.Accuracy()
        m.update_state(y_pred=pred, y_true=self.label_test)
        acc = m.result().numpy()
        print(float(acc))


if __name__ == '__main__':
    path = './flowers/'
    data, label = load_data(path)
    Pred = FlowerClassification(data, label)
