#!/usr/bin/env python3

# 64c-64c-p-128c-128c-p-(256c)4-p-(512c)4-p-(512c)4-p-1560fc-1560fc

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPool2D, Input, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg19 import VGG19

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

def vgg19(input_shape, pretrained_weights=None, use_sigmoid=False):
    # inputs = Input(shape=input_shape)
    #
    # # convolutional layers
    # x = convs(inputs, num_filters=64, kernel_size=3, stride=1, num_layers=2)
    # x = convs(x, num_filters=128, kernel_size=3, stride=1, num_layers=2)
    # x = convs(x, num_filters=256, kernel_size=3, stride=1, num_layers=4)
    # x = convs(x, num_filters=512, kernel_size=3, stride=1, num_layers=4)
    # x = convs(x, num_filters=512, kernel_size=3, stride=1, num_layers=4)

    base_model = VGG19(weights='imagenet')
    inputs = base_model.inputs
    x = base_model.output

    # fully connected layers
    x = GlobalAveragePooling2D()(x)
    # x = Flatten(name="flatten")(x)
    x = Dense(1560, activation="relu")(x)
    x = Dense(1560, activation="relu")(x)
    if use_sigmoid:
        assert input_shape[0] == input_shape[1], "Currently only support equal width and height"
        outputs = Dense(4, activation="sigmoid")(x)
        outputs = keras.layers.Lambda(lambda x: x * input_shape[0])(outputs)
    else:
        outputs = Dense(4, activation="linear")(x)

    # create model
    model = Model(inputs=inputs, outputs=outputs)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights, by_name=True)

    for i in range(len(base_model.layers)):
        base_model.layers[i].trainable = False

    return model
