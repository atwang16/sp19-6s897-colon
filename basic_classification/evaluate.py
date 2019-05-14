

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras import metrics
import numpy as np
import os
from os.path import join
import argparse
from keras.preprocessing.image import load_img,img_to_array

import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def preprocessing_func(X):
    return (X - np.mean(X)) / np.std(X)

# Loading arguments
parser = argparse.ArgumentParser(description='Polyp Detecting Model')

# Model hyperparameters
parser.add_argument('--type', type=str, default='vgg19', help='Determines which convolutional model to use. Valid options are {vgg19|resnet50|pvgg19}')
parser.add_argument('--load_model', type=str, default='saved_models/vgg19_runs/final_model.h5', help='Filepath of a model that will be loaded', required=True)

parser.add_argument('--base_path', default="/basic_classification")
parser.add_argument('--test_path', type=str, default='data/kvasir_train_test_split/test', help='folder that contains all test images')
parser.add_argument('--batch_size', default=16, help='Batch size')


print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()
args.test_path = '/'.join([args.base_path, args.test_path])
args.load_model = '/'.join([args.base_path, args.load_model])

print('\n=== Setting Up Data ===\n')

test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size, class_mode='binary')

print('\n=== Initiating Model ===\n')

model = load_model(args.load_model)
model.compile(optimizer=adam, loss=args.loss, metrics=['accuracy', auc])

model.evaluate_generator(test_generator, 
		steps_per_epoch=train_generator.samples//batch_size, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//batch_size, 
        epochs=epochs)

print(model.metrics_names)
