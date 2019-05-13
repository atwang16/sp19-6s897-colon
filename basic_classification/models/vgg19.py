#!/usr/bin/env python3

# 64c-64c-p-128c-128c-p-(256c)4-p-(512c)4-p-(512c)4-p-1560fc-1560fc

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPool2D, Input, Flatten
from keras.models import Model

def convs(input, num_filters, kernel_size, stride, num_layers=1):
    x = input
    for i in range(num_layers):
        x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=stride,
               padding="same",
               kernel_initializer="he_normal")(x)
    x = MaxPool2D(pool_size=2)(x)
    return x

def vgg(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # convolutional layers
    x = convs(inputs, num_filters=64, kernel_size=3, stride=1, num_layers=2)
    x = convs(x, num_filters=128, kernel_size=3, stride=1, num_layers=2)
    x = convs(x, num_filters=256, kernel_size=3, stride=1, num_layers=4)
    x = convs(x, num_filters=512, kernel_size=3, stride=1, num_layers=4)
    x = convs(x, num_filters=512, kernel_size=3, stride=1, num_layers=4)

    # fully connected layers
    x = Flatten(name="flatten")(x)
    x = Dense(1560, activation="relu")(x)
    x = Dense(1560, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # create model
    model = Model(inputs=inputs, outputs=outputs)

    return model
