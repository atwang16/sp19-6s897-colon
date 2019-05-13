import sklearn.metrics as skm
import numpy as np

import patch_dataset as data
from keras.models import load_model

from roc_callback import auc_metric

import argparse

import os

from keras import backend as K


#python3 patch_majority_voting.py --images data/segmentation/test/polyps/ --ground_truth data/segmentation/test/segmentations/ --load_model PATCH_balanced_mediumvgg_lr_001_trP_70_vP_10/

parser = argparse.ArgumentParser(description='Polyp Detecting Model Evalutaion')
# data location
parser.add_argument('--images', type=str, default='ETIS-LaribPolypDB/ETIS-LaribPolypDB/', help='folder that contains all images that the model will be trained on')
parser.add_argument('--ground_truth', type=str, default='ETIS-LaribPolypDB/GroundTruth/', help='folder that contains all images that contain the ground truth files')

parser.add_argument('--patch_size', type=int, default=32, help='Number of pixels per side in the patch')
parser.add_argument('--batch_size', type=int, default=1, help='Number of pixels per side in the patch')

parser.add_argument('--load_model', type=str, default=None, help='Name of a model that will be loaded')

gpu_devices = K.tensorflow_backend._get_available_gpus()
print("GPU devices: ",gpu_devices)
print("----------------")
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

args = parser.parse_args()

dataset = data.Dataset(args.patch_size, args.images, args.ground_truth)

model = load_model(args.load_model, custom_objects={'auc_metric': auc_metric})

ground_truth_files = os.listdir(args.ground_truth)
original_image_files = os.listdir(args.images)

ground_truth_files = sorted(ground_truth_files)
original_image_files = sorted(original_image_files)

avg_fp = 0
avg_fn = 0

for threshold in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    predicted_labels = []
    for i in range(len(ground_truth_files)):
        ground_truth_name = args.ground_truth+'/' + ground_truth_files[i]
        original_name = args.images +'/' + original_image_files[i]

        img_patches, img_labels = dataset.image_to_sequential_patches(original_name, ground_truth_name)

        batch_size = 100
        false_positive_patches = 0
        false_negative_patches = 0

        pos_predictions = 0
        for i in range(0,len(img_patches),batch_size):
            batch_img_patches = np.array(img_patches[i:i+batch_size])
            batch_img_labels = np.array(img_labels[i:i+batch_size])

            predictions = model.predict(np.array(batch_img_patches))

            false_positive_patches = np.sum(batch_img_labels[:,1] - predictions[:,1] < 0)
            false_negative_patches = np.sum(batch_img_labels[:,1] - predictions[:,1] > 0)

            pos_predictions += np.sum(predictions)

        false_negative_patches = false_negative_patches/len(img_patches)
        false_positive_patches = false_positive_patches/len(img_patches)

        avg_fp = (avg_fp*i + false_positive_patches)/(i+1)
        avg_fn = (avg_fn*i + false_negative_patches)/(i+1)

        percent_polyp = pos_predictions/len(predictions)
        if percent_polyp > threshold:
            img_label = 1
        else:
            img_label = 0
        # print(i, img_label, 'F+',false_positive_patches, 'F-',false_negative_patches)
        predicted_labels.append(img_label)
    print('Threshold',threshold)

    print('Full Image False Negative Rate', (len(predicted_labels) - np.sum(predicted_labels))/len(predicted_labels))

    print('Full Dataset F+',avg_fp)
    print('Full Dataset F-',avg_fn)

    print('AUC',skm.auc(np.ones(len(ground_truth_files)),predicted_labels))
