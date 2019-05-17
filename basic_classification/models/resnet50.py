#!/usr/bin/env python3

# 64c-bn-p-Cb-(Ib)2-Cb-(Ib)3-Cb-(Ib)5-Cb-(Ib)2-p-1024fc

from keras import models
from keras import layers
from keras.applications.resnet50 import ResNet50

def resnet50(input_shape, num_classes=2):

    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')

    x = layers.Dense(1024, activation='relu')(model.layers[-1].output)
    output = layers.Dense(num_classes, activation='softmax')(x)

    new_model = models.Model(input=model.input, output=output)
    return new_model

# if __name__ == '__main__':
#     model = resnet50((224, 224, 3), 2)
#     model.summary()
