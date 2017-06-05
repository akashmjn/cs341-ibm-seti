# Training a CNN on VGG image activations
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD,RMSprop
from keras.callbacks import *
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
K.set_image_dim_ordering('tf')

import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import re
import collections
import os,sys
import sklearn
import commonutils as cu 
from sklearn import svm
from sklearn.externals import joblib

def columnNormalize(data):
    data = data - np.median(data,axis=0)
    #data = data.clip(min=0)
    return data

def evaluateSavedModel(modelPath,dataset,mode):
    """
    mode is either 'test' or 'val' and evaluates on appropriate dataset
    """
    if mode not in ['test','val']: raise ValueError("mode must be 'train/val'")
    best_model = keras.models.load_model(modelPath)
    print("Loaded in model..")
    model_prediction = np.argmax(best_model.predict(dataset["x_{}".format(mode)]), axis=1 )
    print("\nPrinting results on %s dataset for best saved model: \n" % mode)
    y_true = dataset["y_{}".format(mode)]
    temp = classification_report(y_true,model_prediction)
    print(temp)
    temp = confusion_matrix(y_true,model_prediction)
    print(temp)
    print("Test accuracy: %0.2f " % sklearn.metrics.accuracy_score(
        y_true,model_prediction))
    return model_prediction


