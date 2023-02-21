#!usr/bin/python3
# -*- coding: UTF-8 -*-
# Author: Ron
# Date: 2022.11.15 8:51
# latest version: 1.2
# Software: Pycharm

import tensorflow as tf

# set alpha value
alpha = 0.8


# change data type of the fft input
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.complex64)
    y_pred = tf.cast(y_pred, tf.complex64)

    return y_true, y_pred

# F1
# MSE loss function in frequency domain for the speech enhancement with tensorflow
def loss_function(y_true, y_pred):
    y_true, y_pred = weighted_binary_crossentropy(y_true, y_pred)
    # convert to frequency domain
    y_true_fft = tf.signal.fft(y_true)
    y_pred_fft = tf.signal.fft(y_pred)

    # calculate the MSE loss
    loss = tf.reduce_mean(tf.square(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))

    return loss

# T1
# MSE loss function in time domain for the speech enhancement with tensorflow
def loss_function_2(y_true, y_pred):
    # calculate the MSE loss
    loss = tf.reduce_mean(tf.square(y_true - y_pred))

    return loss


# F2
# LAE loss function in frequency domain for the speech enhancement with tensorflow
def loss_function_3(y_true, y_pred):
    y_true, y_pred = weighted_binary_crossentropy(y_true, y_pred)
    # convert to frequency domain
    y_true_fft = tf.signal.fft(y_true)
    y_pred_fft = tf.signal.fft(y_pred)

    # calculate the LAE loss
    loss = tf.reduce_mean(tf.abs(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))

    return loss


# T2
# LAE loss function in time domain for the speech enhancement with tensorflow
def loss_function_4(y_true, y_pred):
    # calculate the LAE loss
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return loss


# T1F1
# create a combined loss function for the speech enhancement with tensorflow
def loss_function_5(y_true, y_pred):
    # calculate the MSE fre loss
    loss_1 = loss_function(y_true, y_pred)

    # calculate the MSE time loss
    loss_2 = loss_function_2(y_true, y_pred)

    # calculate the combined loss
    loss = alpha * loss_1 + (1 - alpha) * loss_2

    return loss


# T2F2
# create a combined loss function for the speech enhancement with tensorflow
def loss_function_6(y_true, y_pred):
    # calculate the LAE fre loss
    loss_1 = loss_function_3(y_true, y_pred)

    # calculate the LAE time loss
    loss_2 = loss_function_4(y_true, y_pred)

    # calculate the combined loss
    loss = alpha * loss_1 + (1 - alpha) * loss_2

    return loss


# T2F1
# create a combined loss function for the speech enhancement with tensorflow
def loss_function_7(y_true, y_pred):
    # calculate the MSE fre loss
    loss_1 = loss_function(y_true, y_pred)

    # calculate the LAE time loss
    loss_2 = loss_function_4(y_true, y_pred)

    # calculate the combined loss
    loss = alpha * loss_1 + (1 - alpha) * loss_2

    return loss


# T1F2
# create a combined loss function for the speech enhancement with tensorflow
def loss_function_8(y_true, y_pred):
    # calculate the LAE fre loss
    loss_1 = loss_function_3(y_true, y_pred)

    # calculate the MSE time loss
    loss_2 = loss_function_2(y_true, y_pred)

    # calculate the combined loss
    loss = alpha * loss_1 + (1 - alpha) * loss_2

    return loss


