import sklearn.metrics as skm
import numpy as np

import patch_dataset as data
from keras.models import load_model

from roc_callback import auc_metric

import argparse

import os

parser = argparse.ArgumentParser(description='Polyp Detecting Model Evalutaion')
# data location
parser.add_argument('--images', type=str, default='ETIS-LaribPolypDB/ETIS-LaribPolypDB/', help='folder that contains all images that the model will be trained on')
parser.add_argument('--ground_truth', type=str, default='ETIS-LaribPolypDB/GroundTruth/', help='folder that contains all images that contain the ground truth files')

parser.add_argument('--patch_size', type=int, default=32, help='Number of pixels per side in the patch')
parser.add_argument('--batch_size', type=int, default=1, help='Number of pixels per side in the patch')

parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

args = parser.parse_args()

dataset = data.Generator_Dataset_Rotated(args.patch_size, args.images, args.ground_truth, batch_size=args.batch_size)

model = load_model(args.load_model, custom_objects={'auc_metric': auc_metric})

ground_truth_files = os.listdir(args.ground_truth)
original_image_files = os.listdir(args.images)

ground_truth_files = sorted(ground_truth_files)
original_image_files = sorted(original_image_files)

for threshold in range(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    print('Threshold',threshold)
    predicted_labels = []
    for i in range(len(ground_truth_files)):
        ground_truth_name = self.ground_truth_location+'/' + ground_truth_files[i]
        original_name = self.original_image_location +'/' + original_image_files[i]

        img_patches, img_labels = datset.image_to_sequential_patches(original_name, ground_truth)

        predictions = model.predict(np.array(img_patches))

        percent_polyp = np.sum(predictions)/len(predictions)

        if percent_polyp > threshold:
            img_label = 1
        else:
            img_label = 0

        predicted_labels.append(img_label)

    print('Full Image False Negative Rate', (len(predicted_labels) - np.sum(predicted_labels))/len(predicted_labels))

    print('AUC',skm.auc(np.ones(len(ground_truth_files)),predicted_labels))
