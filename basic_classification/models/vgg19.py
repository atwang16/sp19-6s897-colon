#!/usr/bin/env python3

# 64c-64c-p-128c-128c-p-(256c)4-p-(512c)4-p-(512c)4-p-1560fc-1560fc

from keras import models
from keras import layers
from keras.applications.vgg19 import VGG19

def vgg19(input_shape, num_classes):

    # convolutional layers
    vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # for layer in vgg_conv.layers:
    #   print(layer, layer.trainable)

    # fully connected layers
    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(1560, activation='relu'))
    model.add(layers.Dense(1560, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
    
