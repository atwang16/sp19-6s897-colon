#!/usr/bin/env python3

# 64c-bn-p-Cb-(Ib)2-Cb-(Ib)3-Cb-(Ib)5-Cb-(Ib)2-p-1024fc

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, MaxPool2D, Input, Flatten
from keras.models import Model

EXPANSION = 4


def conv3x3(inputs, out_planes, strides=1):
    """3x3 convolution with padding"""
    x = Conv2D(out_planes, kernel_size=3, strides=strides, padding="same",
               use_bias=False, kernel_initializer='he_normal',)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def conv1x1(inputs, out_planes, strides=1, activation="relu"):
    """1x1 convolution"""
    x = Conv2D(out_planes, kernel_size=1, strides=strides, use_bias=False, kernel_initializer='he_normal',)(inputs)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def bottleneck(inputs, num_filters, strides=1, downsample=False):
    x = conv1x1(inputs, num_filters)
    x = conv3x3(x, num_filters, strides=strides)
    x = conv1x1(x, num_filters * EXPANSION, activation=None)
    if downsample:
        inputs = conv1x1(inputs, num_filters * EXPANSION, strides, activation=None)
    x = keras.layers.add([x, inputs])
    x = Activation("relu")(x)
    return x


def resnet_layer(inputs, num_filters, num_blocks, strides=1):
    x = bottleneck(inputs, num_filters, strides=strides, downsample=True)
    for _ in range(1, num_blocks):
        x = bottleneck(x, num_filters, strides=1)
    return x


def resnet(input_shape, layers):
    num_filters = 64

    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal',
               use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    for i, num_blocks in enumerate(layers):
        x = resnet_layer(x, num_filters, num_blocks, strides=1 if i == 0 else 2)
        num_filters *= 2

    # Fully connected
    x = AveragePooling2D(pool_size=1)(x)
    y = Flatten()(x)
    y = Dense(1024, kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    outputs = Dense(4,
                    activation="linear",
                    kernel_initializer="he_normal")(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

    return model

def resnet50(input_shape):
    return resnet(input_shape, [3, 4, 6, 3])