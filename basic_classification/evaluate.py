
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras import metrics
import numpy as np
import os
from os.path import join
import argparse
from keras.preprocessing.image import load_img,img_to_array
from keras import optimizers

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def preprocessing_func(X):
    return (X - np.mean(X)) / np.std(X)

# Loading arguments
parser = argparse.ArgumentParser(description='Polyp Detecting Model')

# Model hyperparameters
parser.add_argument('--load_model', type=str, default='saved_models/resnet_runs/model.h5', help='Filepath of a model that will be loaded')

parser.add_argument('--base_path', default="/basic_classification")
parser.add_argument('--test_path', type=str, default='data/kvasir_train_test_split/test', help='folder that contains all test images')


print('\n=== Setting up Parameters ===\n')

args = parser.parse_args()
args.test_path = '/'.join([args.base_path, args.test_path])
args.load_model = '/'.join([args.base_path, args.load_model])

print('\n=== Setting Up Data ===\n')

test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
test_generator = test_datagen.flow_from_directory(args.test_path, target_size=(224, 224), batch_size=1, class_mode='binary', shuffle=False)

print('\n=== Initiating Model ===\n')

model = load_model(args.load_model)

print('\n=== Calculating Evaluation Metrics ===\n')

y_true = test_generator.classes
y_pred = []
y_pred_class = []
for i in range(len(y_true)):
	test_x = next(test_generator)[0]
	y = model.predict(test_x, batch_size=1)
	y_pred.append(y[0][1])
	y_pred_class.append(round(y[0][1]))
	if round(y[0][1]) != y_true[i]:
		print(i)
	
print("Predicted probabilities of containing polyp: ", y_pred)
print("----")
print("Predicted classes: ", y_pred_class)
print("----")
print("Actual classes: ", y_true)
print("----")

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
print(f"True positives = {tp}, false postives = {fp}, false negatives = {fn}, true negatives = {tn}\n")
print(f"PPV: {ppv}, NPV: {npv}\n")
print("AUC score: ", roc_auc_score(y_true, y_pred))



# END FILE