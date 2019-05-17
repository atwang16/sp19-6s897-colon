#!/usr/bin/env python3

# 64c-64c-p-128c-128c-p-(256c)4-p-(512c)4-p-(512c)4-p-1560fc-1560fc

from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16

def vgg16(input_shape, num_classes):

    # convolutional layers
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # fully connected layers
    x = layers.Flatten()(model.layers[-1].output)
    x = layers.Dense(1560, activation='relu', name='fc1')(x)
    x = layers.Dense(1560, activation='relu', name='fc2')(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    new_model = models.Model(input=model.input, outputs=output)
    return new_model

# if __name__ == '__main__':
#     model = vgg16((224, 224, 3), 2)
#     model.summary()
#     pass
