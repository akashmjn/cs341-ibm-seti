import os,sys
import pickle
sys.path.append('/home/cs341seti/cs341-ibm-seti/')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from PIL import Image
import re
import collections
import sklearn
import commonutils as cu
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import classification_report,confusion_matrix

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

""" 
Given a 2-d spectrogram, uses a dynamic programming apprpach to minimize
a loss for finding points. 
Returns [trace,intensitities,minLoss]
"""
def pathTrace(spectrogram,alpha,relativeSearchWindow=0.2):

    # Initializations
    spectrogram = np.squeeze(spectrogram)
    (nrow,ncol) = spectrogram.shape[0:2]
    nSearchWindowSize = int(ncol*relativeSearchWindow) # speed up by not looking very far current point
    lossMat = np.zeros((nrow,ncol))
    pathMat = np.zeros((nrow,ncol))
    trace_vector = np.zeros((nrow,))
    intensity_vector = np.zeros((nrow,))

    # to save compute-time create a delta matrix (d_ij = (f_i-f_j)^2)
    freqs = np.array(range(ncol))
    deltaMat = np.square(freqs[:,np.newaxis]-freqs)

    # For each time instant
    for t in range(nrow):
        # Initialize current row loss
        lossMat[t,:] = -alpha*spectrogram[t,:]
        # if first row, skip the search for best previous point
        if t==0: continue
        # for each frequency for rows 1 (second row) - end
        for f_current in range(ncol):
            # find best previous point
            #searchRange = range(max(f_current-nSearchWindowSize//2,0),min(f_current+nSearchWindowSize//2,ncol-1))
            loss_increments = lossMat[t-1,:] + (1-alpha)*deltaMat[f_current,:]
            f_best_prev = np.argmin(loss_increments)
            best_loss_increment = loss_increments[f_best_prev]
#             best_loss_increment = sys.float_info.max
#             f_best_prev = 0
#             searchRange = (max(f_current-nSearchWindowSize//2,0),min(f_current+nSearchWindowSize//2,ncol-1))
#             for f_prev in range(searchRange[0],searchRange[1]):
#                 loss_increment = lossMat[t-1,f_prev] + (1-alpha)*(f_prev-f_current)**2;
#                 if loss_increment < best_loss_increment:
#                     f_best_prev = f_prev
#                     best_loss_increment = loss_increment
            # update loss for f_current
            lossMat[t,f_current] += best_loss_increment
            # update locally chosen path for f_current
            pathMat[t,f_current] = f_best_prev

    # Finding the best last point and tracing back
    minLoss = np.min(lossMat[nrow-1,:])
    trace_vector[nrow-1] = np.argmin(lossMat[nrow-1,:])
    intensity_vector[nrow-1] = spectrogram[nrow-1,int(trace_vector[nrow-1])]
    # tracing back from end-1 to 0
    for t in range(nrow-2,-1,-1):
        trace_vector[t] = pathMat[t+1,int(trace_vector[t+1])]
        intensity_vector[t] = spectrogram[t,int(trace_vector[t])]

    return [trace_vector,intensity_vector,minLoss]  

class pathTraceLoader:
    def __init__(self,alpha,shape):
        self.alpha = alpha
        self.shape = shape

    def loaderFn(self,file_name):
        x = img_to_array(load_img(file_name,grayscale=True))
        [trace,intensity,minLoss] = pathTrace(cu.modelutils.columnNormalize(x),alpha=self.alpha)
        return np.dstack([trace,intensity])

