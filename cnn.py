#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Create Time : 2021/8/28 12:11 
@Author : Gabriel
@File : cnn.py 
@Project:ImageClassification
@About : 
'''
import tensorflow as tf
from tensorflow.keras import layers


def net_model(num_classes):
    model = tf.keras.Sequential([
        layers.BatchNormalization(),
        layers.Conv2D(8, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], padding='same'),

        layers.Conv2D(16, kernel_size=[1, 1], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], padding='same'),

        layers.Conv2D(16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], padding='same'),

        layers.Conv2D(16, kernel_size=[1, 1], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], padding='same'),

        layers.Flatten(),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dense(num_classes, activation=tf.nn.softmax)

    ])

    return model


def cnnnet(num_classes):
    return net_model(num_classes)
