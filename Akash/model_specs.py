# Defining models to be used for training 
from __future__ import print_function
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD,RMSprop
from keras.callbacks import *
from keras.utils import np_utils
import matplotlib.pyplot as plt
#import tensorflow as tf
import keras
import numpy as np
import collections
import os,sys
import sklearn
import commonutils as cu 


### Fully connected model built on VGG activations 
class fc_1024_256_256:
    @staticmethod
    def build(inputSize,nb_classes,weightsPath=None):
        # Deeper FC classifier network (lesser params) trained on activations
        model = Sequential()
        model.add(Dense(1024,input_shape=(inputSize,),
            activation='relu',init="he_normal"))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Dense(256,activation='relu',init="he_normal"))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Dense(256,activation='relu',init="he_normal"))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(nb_classes,activation='softmax'))
        if weightsPath: model.load_weights(weightsPath)
        return model

### VGG model setup for fine-tuning
class vgg_fine_tune:
    @staticmethod
    def build(input_shape,nb_classes,weightsPath):
        base_model = VGG16(input_shape=input_shape,weights='imagenet',include_top=False)
        flat_layer = Flatten()(base_model.output)
        flatSize = np.prod(base_model.output_shape[1:])
        top_model = fc_1024_256_256.build(flatSize,nb_classes,weightsPath)
        preds = top_model(flat_layer)
        model = Model(inputs = base_model.input, outputs = preds)
        for layer in model.layers[:15]:
            layer.trainable = False
        print(model.summary())
        return model

### VGG-hybrid model directly trained on images
class setiNet_b3_le5:
    @staticmethod
    def build(width,height,depth,nb_classes,weightsPath=None):
        base_model = VGG16(weights='imagenet',include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
#       model.add


## FC classifier network  trained on activations
#model_class = Sequential()
#model_class.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu',init="he_normal"))
#model_class.add(BatchNormalization())
#model_class.add(Dropout(0.5))
#model_class.add(Dense(256,activation='relu',init="he_normal"))
#model_class.add(BatchNormalization())
#model_class.add(Dropout(0.5))
#model_class.add(Dense(nb_classes,activation='softmax'))
#
## FC regression network  trained on activations
#model_reg = Sequential()
#model_reg.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu',init="he_normal"))
#model_reg.add(BatchNormalization())
#model_reg.add(Dropout(0.5))
#model_reg.add(Dense(256,activation='relu'))
#model_reg.add(BatchNormalization())
#model_reg.add(Dropout(0.5))
#model_reg.add(Dense(1))
#
## FC regression network  trained on activations
#model_reg_2048 = Sequential()
#model_reg_2048.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu'))
#model_reg_2048.add(BatchNormalization())
#model_reg_2048.add(Dropout(0.6))
#model_reg_2048.add(Dense(256,activation='relu'))
#model_reg_2048.add(BatchNormalization())
#model_reg_2048.add(Dropout(0.5))
#model_reg_2048.add(Dense(1))
#
## FC regression network  trained on activations
#model_reg_1024d = Sequential()
#model_reg_1024d.add(Dense(1024,input_shape=(X_train.shape[1],),activation='relu'))
#model_reg_1024d.add(BatchNormalization())
#model_reg_1024d.add(Dropout(0.5))
#model_reg_1024d.add(Dense(256,activation='relu'))
#model_reg_1024d.add(BatchNormalization())
#model_reg_1024d.add(Dropout(0.5))
#model_reg_1024d.add(Dense(64,activation='relu'))
#model_reg_1024d.add(BatchNormalization())
#model_reg_1024d.add(Dropout(0.5))
#model_reg_1024d.add(Dense(1))


