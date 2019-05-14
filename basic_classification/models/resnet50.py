#!/usr/bin/env python3

# 64c-bn-p-Cb-(Ib)2-Cb-(Ib)3-Cb-(Ib)5-Cb-(Ib)2-p-1024fc

from keras import models
from keras import layers
from keras.applications.resnet50 import ResNet50

def resnet50(input_shape, num_classes=2):
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')

    model = models.Sequential()
    model.add(resnet)
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model