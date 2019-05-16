#!/usr/bin/env python3

import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import os,datetime
from os.path import join
import pickle
import numpy as np
from matplotlib import pyplot

PATH = os.path.dirname(__file__)
SAVINGS_DIR = join(PATH,'../savings')


def preprocessing_func(X):
    return (X - np.mean(X)) / np.std(X)

def train_model_from_dir(
    base_path, train_path, valid_path, model, model_name='model', target_size=(224, 224), batch_size=16, epochs=500, 
    preprocessing_function=preprocessing_func, params=None
    ):

    SAVINGS_DIR = '/'.join([base_path, 'savings'])

    # Naming and creating folder
    now = datetime.datetime.now()
    model_name = model_name+'_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '-' + str(now.hour) +'h'+ str(now.minute) +'m'+ str(now.second)

    model_name_temp = model_name
    i=0
    while os.path.exists(join(SAVINGS_DIR, model_name_temp)):
        model_name_temp=model_name+'('+str(i)+')'
        i+=1
    MODEL_DIR = join(SAVINGS_DIR, model_name_temp)
    os.makedirs(MODEL_DIR)


    _presaving(model, MODEL_DIR, params)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function, shear_range=0.2, horizontal_flip=True, vertical_flip=True, rotation_range=360)
    train_generator = train_datagen.flow_from_directory(train_path, target_size=target_size, batch_size=batch_size, class_mode='binary', shuffle=True)         
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function, shear_range=0.2, horizontal_flip=True, vertical_flip=True, rotation_range=360)
    validation_generator = train_datagen.flow_from_directory(valid_path, target_size=target_size, batch_size=batch_size, class_mode='binary', shuffle=True)

    tensorboard = TensorBoard(histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    checkpoint = ModelCheckpoint(join(MODEL_DIR,'model.h5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=train_generator.samples//batch_size, 
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//batch_size, 
        epochs=epochs, 
        callbacks=[tensorboard, reduce_lr, checkpoint])

    pyplot.plot(history.history['acc'])
    pyplot.show()

    _postsaving(model, history, MODEL_DIR)

    return model
    
def _presaving(model, model_dir, params):
    with open(join(model_dir,'summary.txt'), 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        print(model.summary())
        sys.stdout = orig_stdout
        f.close()
    with open(join(model_dir,'model.json'), 'w') as f:
        f.write(model.to_json())
    with open(join(model_dir,'params.txt'), 'w') as f:
        f.write(str(params))

def _postsaving(model, history, model_dir):
    model.save_weights(join(model_dir, 'final_model_weights.h5'))
    model.save(join(model_dir, 'final_model.h5'))
    with open(join(model_dir, 'history.pck'), 'wb') as f:
        pickle.dump(history.history, f)
        f.close()




# END FILE