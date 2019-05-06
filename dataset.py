#!/usr/bin/env python3

import sys, os
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image


def load_and_standardize_images(directory, n_samples):

	image_arrays = []
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	if '.DS_Store' in files:
		files.remove('.DS_Store')

	for i in range(n_samples):
		img_path = files[i]
		path = os.path.join(directory, img_path)
		img = image.load_img(path, target_size=(224, 224)) # Resizing the images to (224, 224)
		x = image.img_to_array(img)
		[r, g, b] = np.dsplit(np.array(x), np.array(x).shape[-1])
		r = np.subtract(r, np.mean(r)) / np.std(r)
		g = np.subtract(g, np.mean(g)) / np.std(g)
		b = np.subtract(b, np.mean(b)) / np.std(b)
		img = np.dstack((r, g, b))
		image_arrays.append(img)

	return np.array(image_arrays)

trainX = load_and_standardize_images('/Users/exiao/MIT/hst.956_final_project/sp19-6s897-colon/CVC-ClinicDB_converted/train/polyp', 612)