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
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
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

### Small CNN model directly trained on images
class setiNet:
    @staticmethod
    def build(input_shape,nb_classes,dropout=0.3,init='he_normal',weightsPath=None):
        model = Sequential()
        model.add(Conv2D(8,(3,3),padding='same',input_shape=input_shape,kernel_initializer=init))
        model.add(BatchNormalization())  
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv1 - 256x128x8
        model.add(Conv2D(8,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv2 - 128x64x8
        model.add(Conv2D(32,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv3 - 64x32x32
        model.add(Conv2D(32,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv4 - 32x16x32
        model.add(Conv2D(64,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv5 - 16x8x64
        model.add(Conv2D(64,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv6 - 8x4x64
        model.add(Conv2D(128,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2)))  # Conv7 - 4x2x128
        model.add(Flatten())
        model.add(Dense(256,activation='relu',kernel_initializer=init))
        model.add(BatchNormalization()) # FC1 - 256
        model.add(Dropout(dropout)) # FC1 - 256
        model.add(Dense(nb_classes,activation='softmax',kernel_initializer=init))
        if weightsPath: model.load_weights(weightsPath)
        print(model.summary())
        return model

### CNN model directly trained on images
### Modified from setiNet with consecutive conv blocks, aggressive pooling
### change of aspect ratio, and a fully conv structure
class setiNet_v2:
    @staticmethod
    def build(input_shape,nb_classes,dropout=0.3,init='he_normal',weightsPath=None):
        model = Sequential()
        model.add(Conv2D(8,(3,3),padding='same',input_shape=input_shape,kernel_initializer=init))
        model.add(BatchNormalization())  
        model.add(Activation('relu'))
        model.add(Conv2D(8,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((4,4),name='block1_pool'))  # Convblock1 - 64x128x8
        model.add(Conv2D(16,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(Conv2D(16,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),name='block2_pool'))  # Convblock2 - 32x64x16
        model.add(Conv2D(32,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(Conv2D(32,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),name='block3_pool'))  # Convblock3 - 16x32x32
        model.add(Conv2D(64,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,2),name='block4_pool'))  # Convblock4 - 8x16x64
        model.add(Conv2D(64,(3,3),padding='same',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2,4),name='block5_pool')) # Convblock5 - 4x4x64
        model.add(Conv2D(128,(4,4),padding='valid',kernel_initializer=init))
        model.add(BatchNormalization()) 
        model.add(Activation('relu',name='block6_activation')) # Convblock6 - 1x1x128 : fully convolutional
        model.add(Flatten())
        model.add(Dense(64,activation='relu',kernel_initializer=init))
        model.add(BatchNormalization()) # FC1 - 64
        model.add(Dropout(dropout)) 
        model.add(Dense(16,activation='relu',kernel_initializer=init))
        model.add(BatchNormalization()) # FC2 - 16
        model.add(Dropout(dropout)) 
        model.add(Dense(nb_classes,activation='softmax',kernel_initializer=init))
        if weightsPath: model.load_weights(weightsPath)
        print(model.summary())
        return model

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


