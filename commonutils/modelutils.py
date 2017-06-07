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

def runLinSVMModel(dataset,C,nDataset,modeltype,printReports=True,gamma=None):
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']
    
    # Scaling training and test data
    means = np.mean(x_train,axis=0)
    stddev = np.std(x_train,axis=0)
    # Preventing zero division
    stddev[stddev<1e-3] = 1
    x_train = (x_train - means)/stddev
    x_test = (x_test - means)/stddev
    
    if modeltype=='linSVM':
        lin_clf = svm.LinearSVC(C=C/nDataset,verbose=True,class_weight='balanced')
        lin_clf.fit(x_train, y_train)
        pred_train = lin_clf.predict(x_train)
        pred_test = lin_clf.predict(x_test)
    elif modeltype=='linSVR':
        lin_clf = svm.LinearSVC(C=C/nDataset,verbose=True)
        lin_clf.fit(x_train, y_train)
        pred_train = np.round(lin_clf.predict(x_train))
        pred_test = np.round(lin_clf.predict(x_test))
    elif modeltype=='rbfSVM':
        lin_clf = svm.SVC(C=C/nDataset,gamma=gamma,verbose=True,class_weight='balanced',
                          decision_function_shape='ovr')
        lin_clf.fit(x_train, y_train)
        pred_train = lin_clf.predict(x_train)
        pred_test = lin_clf.predict(x_test)

    train_report = sklearn.metrics.classification_report(y_train,pred_train)
    test_report = sklearn.metrics.classification_report(y_test,pred_test)

    train_confmat = sklearn.metrics.confusion_matrix(y_train,pred_train)
    test_confmat = sklearn.metrics.confusion_matrix(y_test,pred_test)
    
    if printReports:
        print train_report
        print train_confmat
        print test_report
        print test_confmat

        print("Classification accuracy: %0.2f" % sklearn.metrics.accuracy_score(y_test,pred_test) )
        print("MSE: %0.2f" % np.mean(np.square(y_test - lin_clf.predict(x_test))) )
        print("Predictions correlation: %0.2f") % np.corrcoef(y_test,pred_test,rowvar=0)[0,1]
    
    result = {'lin_clf':lin_clf,'train_report':train_report,'train_confmat':train_confmat,
             'test_report':test_report,'test_confmat':test_confmat,
             'train_score':lin_clf.score(x_train,y_train),
             'test_score':lin_clf.score(x_test,y_test)}
    return result
