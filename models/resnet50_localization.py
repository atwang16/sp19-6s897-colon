#!/usr/bin/env python3

# 64c-bn-p-Cb-(Ib)2-Cb-(Ib)3-Cb-(Ib)5-Cb-(Ib)2-p-1024fc

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, Input, Flatten
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


def resnet(input_shape, layers, pretrained_weights=None, use_sigmoid=False):
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
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if use_sigmoid:
        assert input_shape[0] == input_shape[1], "Currently only support equal width and height"
        outputs = Dense(4, activation="sigmoid")(x)
        outputs = keras.layers.Lambda(lambda x: x * input_shape[0])(outputs)
    else:
        outputs = Dense(4, activation="linear", kernel_initializer="he_normal")(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights, by_name=True)

    return model

def resnet50(input_shape, pretrained_weights=None, use_sigmoid=False):
    return resnet(input_shape, [3, 4, 6, 3], pretrained_weights=pretrained_weights, use_sigmoid=use_sigmoid)