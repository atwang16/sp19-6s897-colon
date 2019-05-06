import cv2
import numpy as np
import os

class Dataset:
    def __init__(self, patch_size, original_images, ground_truth, random=True, num_patches = 1000, new_shape=(224,224)):
        self.patch_size = patch_size
        self.input_shape = (patch_size,patch_size,3)
        self.original_image_location = original_images
        self.ground_truth_location = ground_truth

        self.patches, self.labels = self.process_images(random=random, num_patches = num_patches, new_shape=new_shape)

    def normalize_vector(self,mat):
        norm_factor = np.max(mat)
        return mat/norm_factor

    def image_to_random_patches(self, original_image, ground_truth, num_patches = 1000, new_shape = (224,224)):
        def x_trunc(x_val):
            return max(0,min(x_size,x_val))

        def y_trunc(y_val):
            return max(0,min(y_size,y_val))

        patch_num = 0

        ground_truth_img = cv2.imread(ground_truth,0)
        if new_shape is not None:
            ground_truth_img = cv2.resize(ground_truth_img, new_shape, interpolation = cv2.INTER_CUBIC)
        ground_truth_img = self.normalize_vector(ground_truth_img)

        original_image_img = cv2.imread(original_image)
        if new_shape is not None:
            original_image_img = cv2.resize(original_image_img, new_shape, interpolation = cv2.INTER_CUBIC)
        original_image_img = self.normalize_vector(original_image_img)

        patches = []
        labels = []

        while patch_num != num_patches:

            x_size,y_size = new_shape

            x = np.random.randint(0,x_size)
            y = np.random.randint(0,y_size)

            y_lower = y+self.patch_size
            x_right = x+self.patch_size
            left, upper, right, lower = x, y, x_trunc(x_right), y_trunc(y_lower)

            # this is the case that the bounding box is not patch_size x patch_size
            # we cannot pass this into the CNN
            if x_trunc(x_right) != x_right or y_trunc(y_lower) != y_lower:
                continue

            box = (left, upper, right, lower)

            ground_truth_patch = ground_truth_img[upper:lower, left:right]
            original_image_patch = original_image_img[upper:lower, left:right]


            patch_array = np.array(ground_truth_patch)
            mean_patch_value = patch_array.mean()

            if mean_patch_value >= 0.5:
                label = 1
            else:
                label = 0

            patches.append(original_image_patch)

            class_probs = [0,0]
            class_probs[label] = 1
            labels.append(np.array(class_probs))
            patch_num += 1

        return np.array(patches), np.array(labels)

    def image_to_sequential_patches(self, original_image, ground_truth, new_shape = (224,224)):

        ground_truth_img = cv2.imread(ground_truth,0)
        if new_shape is not None:
            ground_truth_img = cv2.resize(ground_truth_img, new_shape, interpolation = cv2.INTER_CUBIC)
        ground_truth_img = self.normalize_vector(ground_truth_img)

        original_image_img = cv2.imread(original_image)
        if new_shape is not None:
            original_image_img = cv2.resize(original_image_img, new_shape, interpolation = cv2.INTER_CUBIC)
        original_image_img = self.normalize_vector(original_image_img)

        x_size,y_size = ground_truth_img.size
        def x_trunc(x_val):
            return max(0,min(x_size,x_val))

        def y_trunc(y_val):
            return max(0,min(y_size,y_val))

        left, upper, right, lower = 0, 0, self.patch_size, self.patch_size
        box = (left, upper, right, lower)

        patches = []
        labels = []
        for x in range(0,x_size,self.patch_size):
            for y in range(0,y_size,self.patch_size):
                y_lower = y+self.patch_size
                x_right = x+self.patch_size
                left, upper, right, lower = x, y, x_trunc(x_right), y_trunc(y_lower)

                # this is the case that the bounding box is not patch_size x patch_size
                # we cannot pass this into the CNN
                if x_trunc(x_right) != x_right or y_trunc(y_lower) != y_lower:
                    continue

                box = (left, upper, right, lower)

                ground_truth_patch = ground_truth_img[upper:lower, left:right]
                original_image_patch = original_image_img[upper:lower, left:right]


                patch_array = np.array(ground_truth_patch)
                mean_patch_value = patch_array.mean()

                if mean_patch_value >= 0.5:
                    label = 1
                else:
                    label = 0

                patches.append(original_image_patch)

                class_probs = [0,0]
                class_probs[label] = 1
                labels.append(np.array(class_probs))

        return np.array(patches), np.array(labels)

    # assuming they all have the same (or similar names) and are alphabetical
    def process_images(self, random = True, num_patches = 1000, new_shape=(224,224)):
        ground_truth_files = os.listdir(self.ground_truth_location)
        original_image_files = os.listdir(self.original_image_location)

        ground_truth_files = sorted(ground_truth_files)
        original_image_files = sorted(original_image_files)
        patches = []
        labels = []

        for i in range(len(ground_truth_files)):
            # print(i)
            ground_truth_name = self.ground_truth_location + ground_truth_files[i]
            original_name = self.original_image_location + original_image_files[i]

            if random:
                current_patches, current_labels = self.image_to_random_patches(original_name, ground_truth_name, num_patches, new_shape)

            else:
                current_patches, current_labels = self.image_to_sequential_patches(original_name, ground_truth_name, new_shape)

            patches.extend(current_patches)
            labels.extend(current_labels)

        return np.array(patches), np.array(labels)

    def split_data(self, train_percent = 0.1, validation_percent = 0.2):

        if train_percent + validation_percent >= 1:
            # raise ValueError
            return None

        train_number = int(len(self.patches)*train_percent)
        valid_number = int(len(self.patches)*validation_percent)

        training_indices = np.random.choice(np.arange(len(self.patches)), train_number, replace=False)

        remaining_indices = set(np.arange(len(self.patches))).difference(set(training_indices))

        valid_indices = np.random.choice(list(remaining_indices), valid_number, replace=False)

        remaining_indices = remaining_indices.difference(set(valid_indices))

        test_indices = np.array(list(remaining_indices))

        train_patches = []
        for idx in training_indices:
            train_patches.append(np.array(self.patches[idx]))
        train_labels = self.labels[training_indices]

        valid_patches = []
        for idx in valid_indices:
            valid_patches.append(np.array(self.patches[idx]))
        valid_labels = self.labels[valid_indices]

        test_patches = []
        for idx in test_indices:
            test_patches.append(np.array(self.patches[idx]))
        test_labels = self.labels[test_indices]

        return (np.array(train_patches), np.array(train_labels)), (np.array(valid_patches), np.array(valid_labels)), (np.array(test_patches), np.array(test_labels))


# im = Image.open('CVC-ClinicDB/GroundTruth/2.tif')
# data = Dataset(10, 'ETIS-LaribPolypDB/ETIS-LaribPolypDB/', 'ETIS-LaribPolypDB/GroundTruth/',num_patches=1000)
# #
# (train_patches, train_labels), (valid_patches, valid_labels), (test_patches, test_labels) = data.split_data(train_percent=0.5,validation_percent=0)
# import pdb; pdb.set_trace()
# patches = data.process_images('ETIS-LaribPolypDB/ETIS-LaribPolypDB/2.tif','ETIS-LaribPolypDB/GroundTruth/p2.tif')
