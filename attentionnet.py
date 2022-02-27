#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2022/2/27 19:54 
@Author : Gabriel
@File : attentionnet.py 
@Project:ImageClassification
@About : 
'''
import math

import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential

tf.random.set_seed(1234)


def channel_block(inputs, ratio=4):
    '''
    通道注意力
    :param inputs: 输入
    :param ratio: 第一次卷积的压缩比例
    :return:
    '''
    c = inputs.shape[-1]
    share_conv1 = layers.Conv2D(c // ratio, activation='relu')
    share_conv2 = layers.Conv2D(c)

    # [c]
    pool_max = layers.GlobalMaxPool2D()(inputs)
    pool_avg = layers.GlobalAvgPool2D()(inputs)

    # [c]-->[1,1,c]
    pool_max = layers.Reshape([1, 1, -1])(pool_max)
    pool_avg = layers.Reshape([1, 1, -1])(pool_avg)

    pool_max = share_conv1(pool_max)
    pool_max = share_conv2(pool_max)

    pool_avg = share_conv1(pool_avg)
    pool_avg = share_conv2(pool_avg)

    x = layers.Add()([pool_max, pool_avg])
    x = layers.Activation('sigmoid')(x)

    outs = layers.Multiply()([inputs, x])

    return outs


def spatial_attention(inputs):
    '''
    空间注意力
    :param inputs:
    :return:
    '''
    x_max = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    x_avg = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)

    x = layers.Concatenate()([x_max, x_avg])
    x = layers.Conv2D(1, [7, 7], activation='sigmoid')(x)

    outs = layers.Multiply()([inputs, x])

    return outs


def cbem_block(inputs, ratio=4):
    x = channel_block(inputs, ratio)
    x = spatial_attention(x)

    return x


def se_block(inputs, ratio=4):
    # inputs --> [h,w,c]
    c = inputs.shape[-1]

    x = layers.GlobalAveragePooling2D()(inputs)
    # x -->[1,1,-1]
    x = layers.Reshape([1, 1, -1])(x)

    # 第一次全连接
    x = layers.Dense(c // ratio)(x)
    x = layers.Activation('relu')(x)

    # 第二次全连接
    x = layers.Dense(c)(x)
    x = layers.Activation('sigmoid')(x)

    out = layers.Multiply()([x, inputs])

    return out


def eca_block(inputs, b=1, gamma=2):
    # inputs --> [h,w,c]
    c = inputs.shape[-1]
    kernel_size = int(abs(((math.log(c, 2) + b) / gamma)))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    x = layers.GlobalAveragePooling2D()(inputs)
    # x -->[1,1,-1]
    x = layers.Reshape([-1, 1])(x)

    x = layers.Conv1D(1, kernel_size, padding='same', use_bias=False)(x)
    x = layers.Activation('sigmoid')(x)
    x = layers.Reshape([1, 1, -1])(x)
    out = layers.Multiply()([x, inputs])

    return out


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, trainning=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes, attention_type=0):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            layers.Conv2D(64, (32, 32), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])
        self.attention_type = attention_type
        self.attention = [eca_block, se_block, cbem_block]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention[self.attention_type](x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18(num_classes, attention_type):
    return ResNet([2, 2, 2, 2], num_classes=num_classes, attention_type=attention_type)
