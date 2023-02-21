#!usr/bin/python3
# -*- coding: UTF-8 -*-
# Author: Ron
# Date: 2022.11.1 16:51
# Software: Pycharm

import tensorflow as tf


print(tf.__version__)


def simplenet(input_size=(128, 128, 1)):
    # size filter input
    size_filter_in = 8
    # normal initialization of weights
    kernel_init = 'he_normal'
    # To apply leaky relu after the conv layer
    activation_layer = None
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(size_filter_in, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(inputs)
    conv1 = tf.keras.layers.ELU()(conv1)
    conv1 = tf.keras.layers.Conv2D(size_filter_in, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv1)
    conv1 = tf.keras.layers.ELU()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool1)
    conv2 = tf.keras.layers.ELU()(conv2)
    conv2 = tf.keras.layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv2)
    conv2 = tf.keras.layers.ELU()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool2)
    conv3 = tf.keras.layers.ELU()(conv3)
    conv3 = tf.keras.layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv3)
    conv3 = tf.keras.layers.ELU()(conv3)
    conv4 = tf.keras.layers.Conv2D(1, 1, activation='tanh')(conv3)

    model2 = tf.keras.models.Model(inputs, conv4)

    # 定义alpha
    model2.compile(optimizer='adam', loss=tf.losses.mse(), metrics=['mse'])  # 此处设置的损失函数后续会调整为自定义损失函数
