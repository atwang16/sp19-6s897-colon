import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

import imutils

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
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
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
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
        if new_shape is not None:
            original_image_img = cv2.resize(original_image_img, new_shape, interpolation = cv2.INTER_CUBIC)
        original_image_img = self.normalize_vector(original_image_img)

        y_size,x_size = ground_truth_img.shape
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

                # print(x_right,x_trunc(x_right))
                # print(y_lower,y_trunc(y_lower))

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

    def split_data(self, train_percent = 0.1, validation_percent = 0.2, balance_classes=False):

        class_patches = self.patches.copy()
        class_labels = self.labels.copy()

        if train_percent + validation_percent >= 1:
            # raise ValueError
            return None

        if balance_classes:

            pos_idcs = np.where(self.labels[:,1] == 1)
            # num_neg = total_patches - pos_patches
            # class_difference = num_neg - pos_paches
            # class_difference = total_patches - 2*pos_paches
            class_diff = len(class_patches) - 2*len(pos_idcs[0])
            if class_diff < 0:
                return None

            over_sampled_idcs = np.random.choice(pos_idcs[0], class_diff)

            pos_oversampled_idcs = np.concatenate((pos_idcs[0],over_sampled_idcs))

            pos_train_number = int(len(pos_oversampled_idcs)*train_percent)
            pos_valid_number = int(len(pos_oversampled_idcs)*validation_percent)

            # getting relevant idexes from oversample
            train_pos_idcs = np.random.choice(range(len(pos_oversampled_idcs)), pos_train_number, replace=False)

            pos_remaining_indices = set(range(len(pos_oversampled_idcs))).difference(set(train_pos_idcs))

            valid_pos_idcs = np.random.choice(list(pos_remaining_indices), pos_valid_number, replace=False)

            pos_remaining_indices = pos_remaining_indices.difference(set(valid_pos_idcs))

            test_pos_idcs = np.array(list(pos_remaining_indices))

            # getting real indices
            train_pos_idcs = pos_oversampled_idcs[train_pos_idcs]
            valid_pos_idcs = pos_oversampled_idcs[valid_pos_idcs]
            test_pos_idcs = pos_oversampled_idcs[test_pos_idcs]




            neg_train_number = int(len(pos_oversampled_idcs)*train_percent)
            neg_valid_number = int(len(pos_oversampled_idcs)*validation_percent)

            negative_idcs = np.where(class_labels[:,1] == 0)[0]

            train_neg_idcs = np.random.choice(negative_idcs, neg_train_number, replace=False)

            neg_remaining_indices = set(negative_idcs).difference(set(train_neg_idcs))

            valid_neg_idcs = np.random.choice(list(neg_remaining_indices), neg_valid_number, replace=False)

            neg_remaining_indices = neg_remaining_indices.difference(set(valid_neg_idcs))

            test_neg_idcs = np.array(list(neg_remaining_indices))


            training_indices = np.concatenate((train_pos_idcs,train_neg_idcs))
            valid_indices = np.concatenate((valid_pos_idcs,valid_neg_idcs))
            test_indices = np.concatenate((test_pos_idcs,test_neg_idcs))

            # import pdb; pdb.set_trace()


            train_patches = []
            for idx in training_indices:
                angle = 0*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                train_patches.append(np.array(rotated))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                angle = 0*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                valid_patches.append(np.array(rotated))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                angle = 0*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                test_patches.append(np.array(rotated))
            test_labels = class_labels[test_indices]

        else:
            train_number = int(len(class_patches)*train_percent)
            valid_number = int(len(class_patches)*validation_percent)

            training_indices = np.random.choice(np.arange(len(class_patches)), train_number, replace=False)

            remaining_indices = set(np.arange(len(class_patches))).difference(set(training_indices))

            valid_indices = np.random.choice(list(remaining_indices), valid_number, replace=False)

            remaining_indices = remaining_indices.difference(set(valid_indices))

            test_indices = np.array(list(remaining_indices))

            train_patches = []
            for idx in training_indices:
                train_patches.append(np.array(class_patches[idx]))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                valid_patches.append(np.array(class_patches[idx]))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                test_patches.append(np.array(class_patches[idx]))
            test_labels = class_labels[test_indices]

        return (np.array(train_patches), np.array(train_labels)), (np.array(valid_patches), np.array(valid_labels)), (np.array(test_patches), np.array(test_labels))

class Dataset_Regression:
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
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
        if new_shape is not None:
            original_image_img = cv2.resize(original_image_img, new_shape, interpolation = cv2.INTER_CUBIC)
        # original_image_img = self.normalize_vector(original_image_img)

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

            patches.append(original_image_patch)

            labels.append(mean_patch_value)
            patch_num += 1

        return np.array(patches), np.array(labels)

    def image_to_sequential_patches(self, original_image, ground_truth, new_shape = (224,224)):

        ground_truth_img = cv2.imread(ground_truth,0)
        if new_shape is not None:
            ground_truth_img = cv2.resize(ground_truth_img, new_shape, interpolation = cv2.INTER_CUBIC)
        ground_truth_img = self.normalize_vector(ground_truth_img)

        original_image_img = cv2.imread(original_image)
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
        if new_shape is not None:
            original_image_img = cv2.resize(original_image_img, new_shape, interpolation = cv2.INTER_CUBIC)
        original_image_img = self.normalize_vector(original_image_img)

        y_size,x_size = ground_truth_img.shape
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

                # print(x_right,x_trunc(x_right))
                # print(y_lower,y_trunc(y_lower))

                box = (left, upper, right, lower)

                ground_truth_patch = ground_truth_img[upper:lower, left:right]
                original_image_patch = original_image_img[upper:lower, left:right]


                patch_array = np.array(ground_truth_patch)
                mean_patch_value = patch_array.mean()

                patches.append(original_image_patch)

                labels.append(mean_patch_value)
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

    def split_data(self, train_percent = 0.1, validation_percent = 0.2, balance_classes=True):

        class_patches = self.patches.copy()
        class_labels = self.labels.copy()

        if train_percent + validation_percent >= 1:
            # raise ValueError
            return None

        if balance_classes:

            pos_idcs = np.where(self.labels > 0)
            # import pdb; pdb.set_trace()
            # num_neg = total_patches - pos_patches
            # class_difference = num_neg - pos_paches
            # class_difference = total_patches - 2*pos_paches
            class_diff = len(class_patches) - 2*len(pos_idcs[0])
            if class_diff < 0:
                return None

            over_sampled_idcs = np.random.choice(pos_idcs[0], class_diff)

            pos_oversampled_idcs = np.concatenate((pos_idcs[0],over_sampled_idcs))

            pos_train_number = int(len(pos_oversampled_idcs)*train_percent)
            pos_valid_number = int(len(pos_oversampled_idcs)*validation_percent)

            # getting relevant idexes from oversample
            train_pos_idcs = np.random.choice(range(len(pos_oversampled_idcs)), pos_train_number, replace=False)

            pos_remaining_indices = set(range(len(pos_oversampled_idcs))).difference(set(train_pos_idcs))

            valid_pos_idcs = np.random.choice(list(pos_remaining_indices), pos_valid_number, replace=False)

            pos_remaining_indices = pos_remaining_indices.difference(set(valid_pos_idcs))

            test_pos_idcs = np.array(list(pos_remaining_indices))

            # getting real indices
            train_pos_idcs = pos_oversampled_idcs[train_pos_idcs]
            valid_pos_idcs = pos_oversampled_idcs[valid_pos_idcs]
            test_pos_idcs = pos_oversampled_idcs[test_pos_idcs]




            neg_train_number = int(len(pos_oversampled_idcs)*train_percent)
            neg_valid_number = int(len(pos_oversampled_idcs)*validation_percent)

            negative_idcs = np.where(class_labels == 0)[0]

            train_neg_idcs = np.random.choice(negative_idcs, neg_train_number, replace=False)

            neg_remaining_indices = set(negative_idcs).difference(set(train_neg_idcs))

            valid_neg_idcs = np.random.choice(list(neg_remaining_indices), neg_valid_number, replace=False)

            neg_remaining_indices = neg_remaining_indices.difference(set(valid_neg_idcs))

            test_neg_idcs = np.array(list(neg_remaining_indices))


            training_indices = np.concatenate((train_pos_idcs,train_neg_idcs))
            valid_indices = np.concatenate((valid_pos_idcs,valid_neg_idcs))
            test_indices = np.concatenate((test_pos_idcs,test_neg_idcs))

            # import pdb; pdb.set_trace()


            train_patches = []
            for idx in training_indices:
                #import pdb; pdb.set_trace()
                train_patches.append(np.array(class_patches[idx]))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                valid_patches.append(np.array(class_patches[idx]))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                test_patches.append(np.array(class_patches[idx]))
            test_labels = class_labels[test_indices]

        else:
            train_number = int(len(class_patches)*train_percent)
            valid_number = int(len(class_patches)*validation_percent)

            training_indices = np.random.choice(np.arange(len(class_patches)), train_number, replace=False)

            remaining_indices = set(np.arange(len(class_patches))).difference(set(training_indices))

            valid_indices = np.random.choice(list(remaining_indices), valid_number, replace=False)

            remaining_indices = remaining_indices.difference(set(valid_indices))

            test_indices = np.array(list(remaining_indices))

            train_patches = []
            for idx in training_indices:
                train_patches.append(np.array(class_patches[idx]))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                valid_patches.append(np.array(class_patches[idx]))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                test_patches.append(np.array(class_patches[idx]))
            test_labels = class_labels[test_indices]

        return (np.array(train_patches), np.array(train_labels)), (np.array(valid_patches), np.array(valid_labels)), (np.array(test_patches), np.array(test_labels))

    def split_data_unbalanced(self, train_percent = 0.1, validation_percent = 0.2):

        class_patches = self.patches.copy()
        class_labels = self.labels.copy()

        if train_percent + validation_percent >= 1:
            # raise ValueError
            return None

        train_number = int(len(class_patches)*train_percent)
        valid_number = int(len(class_patches)*validation_percent)

        training_indices = np.random.choice(np.arange(len(class_patches)), train_number, replace=False)

        remaining_indices = set(np.arange(len(class_patches))).difference(set(training_indices))

        valid_indices = np.random.choice(list(remaining_indices), valid_number, replace=False)

        remaining_indices = remaining_indices.difference(set(valid_indices))

        test_indices = np.array(list(remaining_indices))

        train_patches = []
        for idx in training_indices:
            train_patches.append(np.array(class_patches[idx]))
        train_labels = class_labels[training_indices]

        valid_patches = []
        for idx in valid_indices:
            valid_patches.append(np.array(class_patches[idx]))
        valid_labels = class_labels[valid_indices]

        test_patches = []
        for idx in test_indices:
            test_patches.append(np.array(class_patches[idx]))
        test_labels = class_labels[test_indices]

        return (np.array(train_patches), np.array(train_labels)), (np.array(valid_patches), np.array(valid_labels)), (np.array(test_patches), np.array(test_labels))

class Dataset_Rotated:
    def __init__(self, patch_size, original_images, ground_truth, random=True, num_patches = 1000, new_shape=(224,224)):
        self.patch_size = patch_size
        self.input_shape = (patch_size,patch_size,3)
        self.original_image_location = original_images
        self.ground_truth_location = ground_truth
        self.patches, self.labels = self.process_images(random=random, num_patches = num_patches, new_shape=new_shape)

    def normalize_vector(self,mat):
        #import pdb; pdb.set_trace()
        mean = np.mean(mat)
        std = np.std(mat)
        return (mat - mean) / std

    def rotation(self,point,center,radians):
        sin = np.sin(radians)
        cos = np.cos(radians)

        x,y = point
        cx,cy = center

        xnew = x*cos - y*sin
        ynew = x*sin + y*cos

        return (xnew + cx, ynew + cy)

    def center_point(self, box):
        (left, upper, right, lower) = box

        cx = (left[0] + right[0])//2
        cy = (upper[1] + lower[1])//2

        return cx,cy

    def image_to_random_patches(self, original_image, ground_truth, num_patches = 1000, new_shape = (224,224)):
        def x_trunc(x_val):
            return max(0,min(x_size,x_val))

        def y_trunc(y_val):
            return max(0,min(y_size,y_val))

        pos_patch_num = 0
        neg_patch_num = 0

        ground_truth_img = cv2.imread(ground_truth,0)
        ground_truth_img = self.normalize_vector(ground_truth_img)

        original_image_img = cv2.imread(original_image)
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
        original_image_img = self.normalize_vector(original_image_img)

        pos_patches = []

        while pos_patch_num != num_patches:
            # print(pos_patch_num)

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

            if label == 1:
                pos_patches.append(original_image_patch)
                pos_patch_num += 1


        neg_patches = []

        while neg_patch_num != num_patches:

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

            if label == 0:
                neg_patches.append(original_image_patch)
                neg_patch_num += 1

        patches = []
        labels = [[0,1]]*num_patches + [[1,0]]*num_patches
        patches = pos_patches
        patches.extend(neg_patches)
        return np.array(patches), np.array(labels)

    def image_to_sequential_patches(self, original_image, ground_truth, new_shape = (224,224)):
        ground_truth_img = cv2.imread(ground_truth,0)
        ground_truth_img = self.normalize_vector(ground_truth_img)

        original_image_img = cv2.imread(original_image)
        original_image_img = cv2.cvtColor(original_image_img, cv2.COLOR_BGR2RGB)
        original_image_img = self.normalize_vector(original_image_img)

        pos_patches = []
        neg_patches = []
        labels = []

        for angle in np.random.choice(np.arange(0, 360),2,replace=False):
            rotated_original = imutils.rotate(original_image_img, angle)

            rotated_gnd = imutils.rotate(ground_truth_img, angle)


            y_size,x_size = ground_truth_img.shape
            def x_trunc(x_val):
                return max(0,min(x_size,x_val))

            def y_trunc(y_val):
                return max(0,min(y_size,y_val))

            left, upper, right, lower = 0, 0, self.patch_size, self.patch_size
            box = (left, upper, right, lower)

            for x in range(0,x_size,self.patch_size):
                for y in range(0,y_size,self.patch_size):
                    y_lower = y+self.patch_size
                    x_right = x+self.patch_size
                    left, upper, right, lower = x, y, x_trunc(x_right), y_trunc(y_lower)

                    # this is the case that the bounding box is not patch_size x patch_size
                    # we cannot pass this into the CNN
                    if x_trunc(x_right) != x_right or y_trunc(y_lower) != y_lower:
                        continue

                    # print(x_right,x_trunc(x_right))
                    # print(y_lower,y_trunc(y_lower))

                    box = (left, upper, right, lower)

                    ground_truth_patch = ground_truth_img[upper:lower, left:right]
                    original_image_patch = original_image_img[upper:lower, left:right]

                    shift_x = int(np.random.rand()*(self.patch_size/2))
                    shift_y = int(np.random.rand()*(self.patch_size/2))

                    # UL
                    UL = original_image_img[y_trunc(upper-shift_y):y_trunc(lower-shift_y), x_trunc(left-shift_x):x_trunc(right-shift_x)]

                    # UC
                    UC = original_image_img[y_trunc(upper-shift_y):y_trunc(lower-shift_y), x_trunc(left):x_trunc(right)]

                    # UR
                    UR = original_image_img[y_trunc(upper-shift_y):y_trunc(lower-shift_y), x_trunc(left+shift_x):x_trunc(right+shift_x)]



                    # LL
                    LL = original_image_img[y_trunc(upper+shift_y):y_trunc(lower+shift_y), x_trunc(left-shift_x):x_trunc(right-shift_x)]

                    # LC
                    LC = original_image_img[y_trunc(upper+shift_y):y_trunc(lower+shift_y), x_trunc(left):x_trunc(right)]

                    # LR
                    LR = original_image_img[y_trunc(upper+shift_y):y_trunc(lower+shift_y), x_trunc(left+shift_x):x_trunc(right+shift_x)]



                    # CL
                    CL = original_image_img[y_trunc(upper):y_trunc(lower), x_trunc(left-shift_x):x_trunc(right-shift_x)]

                    # CR
                    CR = original_image_img[y_trunc(upper):y_trunc(lower), x_trunc(left+shift_x):x_trunc(right+shift_x)]



                    patch_array = np.array(ground_truth_patch)
                    mean_patch_value = patch_array.mean()
                    if mean_patch_value >= 0.75:
                        label = 1

                        if UL.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(UL)

                        if UC.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(UC)
                        if UR.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(UR)
                        if LL.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(LL)

                        if LC.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(LC)

                        if LR.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(LR)

                        if CL.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(CL)

                        if CR.shape == (self.patch_size,self.patch_size,3):
                            pos_patches.append(CR)

                        pos_patches.append(original_image_patch)
                    else:
                        neg_patches.append(original_image_patch)



        # import pdb; pdb.set_trace()
        # for x in patches:
        #     print(x.shape)
        # python3 train_images.py --num_patches 10 --lr 0.001 --type pvgg19_pretrained --output_dir reweighted_balanced_pretrained_categorical/ --train_percent 0.7 --patch_size 32 --loss categorical_crossentropy --random_patches False

        if len(pos_patches) < len(neg_patches):
            random_neg_ids = np.random.choice(range(len(neg_patches)),len(pos_patches),replace=False)
            neg_patches = np.array(neg_patches)[random_neg_ids]
            pos_patches = np.array(pos_patches)
        elif len(pos_patches) >= len(neg_patches):
            random_pos_ids = np.random.choice(range(len(pos_patches)),len(neg_patches),replace=False)
            pos_patches = np.array(pos_patches)[random_pos_ids]
            neg_patches = np.array(neg_patches)

        # print('POS EXAMPLES',len(pos_patches))
        # print('NEG EXAMPLES',len(neg_patches))

        patches = np.vstack((pos_patches,neg_patches))
        # print('LEN LABELS',len(labels))
        # import pdb; pdb.set_trace()
        labels = [[0,1]]*len(pos_patches) + [[1,0]]*len(neg_patches)
        # print('len labels',len(labels))
        return np.array(patches), np.array(labels)

    # assuming they all have the same (or similar names) and are alphabetical
    def process_images(self, random = True, num_patches = 1000, new_shape=(224,224)):
        ground_truth_files = os.listdir(self.ground_truth_location)
        original_image_files = os.listdir(self.original_image_location)
        #import pdb; pdb.set_trace()
        ground_truth_files = sorted(ground_truth_files)
        original_image_files = sorted(original_image_files)
        patches = []
        labels = []

        for i in range(len(ground_truth_files)):
            # print(i)
            ground_truth_name = self.ground_truth_location+'/' + ground_truth_files[i]
            original_name = self.original_image_location +'/' + original_image_files[i]
            #import pdb; pdb.set_trace()
            if random:
                # print('RAND')
                current_patches, current_labels = self.image_to_random_patches(original_name, ground_truth_name, num_patches, new_shape)

            else:
                # print('SEQ')
                current_patches, current_labels = self.image_to_sequential_patches(original_name, ground_truth_name, new_shape)

            # print('SINGLE IMG')
            # import pdb; pdb.set_trace()
            patches.extend(current_patches)
            labels.extend(current_labels)
        # print('PROCES IMGAES')
        # import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        return np.array(patches), np.array(labels)

    def split_data(self, train_percent = 0.1, validation_percent = 0.2, balance_classes=False):

        class_patches = self.patches.copy()
        class_labels = self.labels.copy()

        if train_percent + validation_percent >= 1:
            # raise ValueError
            return None

        if balance_classes:

            pos_idcs = np.where(self.labels[:,1] == 1)[0]
            # num_neg = total_patches - pos_patches
            # class_difference = num_neg - pos_paches
            # class_difference = total_patches - 2*pos_paches

            train_number = int(len(pos_idcs)*train_percent)
            valid_number = int(len(pos_idcs)*validation_percent)

            # getting relevant idexes from oversample
            train_pos_idcs = np.random.choice(range(len(pos_idcs)), train_number, replace=False)

            pos_remaining_indices = set(range(len(pos_idcs))).difference(set(train_pos_idcs))

            valid_pos_idcs = np.random.choice(list(pos_remaining_indices), valid_number, replace=False)

            pos_remaining_indices = pos_remaining_indices.difference(set(valid_pos_idcs))

            test_pos_idcs = np.array(list(pos_remaining_indices))

            # getting real indices
            train_pos_idcs = pos_idcs[train_pos_idcs]
            valid_pos_idcs = pos_idcs[valid_pos_idcs]
            test_pos_idcs = pos_idcs[test_pos_idcs]

            negative_idcs = np.where(class_labels[:,1] == 0)[0]

            train_neg_idcs = np.random.choice(negative_idcs, train_number, replace=False)

            neg_remaining_indices = set(negative_idcs).difference(set(train_neg_idcs))

            valid_neg_idcs = np.random.choice(list(neg_remaining_indices), valid_number, replace=False)

            neg_remaining_indices = neg_remaining_indices.difference(set(valid_neg_idcs))

            test_neg_idcs = np.array(list(neg_remaining_indices))


            training_indices = np.concatenate((train_pos_idcs,train_neg_idcs))
            valid_indices = np.concatenate((valid_pos_idcs,valid_neg_idcs))
            test_indices = np.concatenate((test_pos_idcs,test_neg_idcs))

            train_patches = []
            for idx in training_indices:
                angle = 0*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                train_patches.append(np.array(rotated))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                angle = 360*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                valid_patches.append(np.array(rotated))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                angle = 360*np.random.rand(1).item()
                rotated = imutils.rotate(class_patches[idx], angle)
                test_patches.append(np.array(rotated))
            test_labels = class_labels[test_indices]

        else:
            train_number = int(len(class_patches)*train_percent)
            valid_number = int(len(class_patches)*validation_percent)

            training_indices = np.random.choice(np.arange(len(class_patches)), train_number, replace=False)

            remaining_indices = set(np.arange(len(class_patches))).difference(set(training_indices))

            valid_indices = np.random.choice(list(remaining_indices), valid_number, replace=False)

            remaining_indices = remaining_indices.difference(set(valid_indices))

            test_indices = np.array(list(remaining_indices))

            train_patches = []
            for idx in training_indices:
                train_patches.append(np.array(class_patches[idx]))
            train_labels = class_labels[training_indices]

            valid_patches = []
            for idx in valid_indices:
                valid_patches.append(np.array(class_patches[idx]))
            valid_labels = class_labels[valid_indices]

            test_patches = []
            for idx in test_indices:
                test_patches.append(np.array(class_patches[idx]))
            test_labels = class_labels[test_indices]

        return (np.array(train_patches), np.array(train_labels)), (np.array(valid_patches), np.array(valid_labels)), (np.array(test_patches), np.array(test_labels))

# # im = Image.open('CVC-ClinicDB/GroundTruth/2.tif')
# data = Dataset(10, 'ETIS-LaribPolypDB/ETIS-LaribPolypDB/', 'ETIS-LaribPolypDB/GroundTruth/',num_patches=1)
# # #
# (train_patches, train_labels), (valid_patches, valid_labels), (test_patches, test_labels) = data.split_data(train_percent=0.5,validation_percent=0,balance_classes=True)
# # import pdb; pdb.set_trace()
# # patches = data.process_images('ETIS-LaribPolypDB/ETIS-LaribPolypDB/2.tif','ETIS-LaribPolypDB/GroundTruth/p2.tif')
