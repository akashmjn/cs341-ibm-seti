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
#import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import re
import collections
import os,sys
import sklearn
import commonutils as cu 
from sklearn import svm
from sklearn.externals import joblib

# Taking in command line args
# python activationModel.py MODELTYPE NUM_TRAIN NEpochs OPTIM LR DECAY subset
args = sys.argv

batch_size = 75
model_type = args[1]
#num_train = int(args[2])
nb_epoch = int(args[3])
optim = args[4]
lr = float(args[5])
decay = float(args[6])

data_augmentation = False
# creating dir to save results
os.system("mkdir -p plots")
os.system("mkdir -p savedModels")

# Loading either full dataset or subset of classes
if len(args)==8:
    subsetClasses = {0.0:0.0,2.0:1.0,3.0:2.0,5.0:3.0}
    dataset = cu.datautils.loadDataset("data/activations-4-19.h5", subsetClasses=subsetClasses)
    nb_classes = 4
else:
    nb_classes = 7
    dataset = cu.datautils.loadDataset("data/activations-4-19.h5")

modelName = '{}class_{}lr{}decay{}'.format(nb_classes,optim,lr,decay)
num_val = dataset['x_val'].shape[0]
num_test = dataset['x_test'].shape[0]
num_train = dataset['x_train'].shape[0]

# Creating datasets
X_train = dataset['x_train']
y_train = dataset['y_train']
X_val = dataset['x_val']
y_val = dataset['y_val']
X_test = dataset['x_test']
y_test = dataset['y_test']
## Scaling training and test data
#means = np.mean(x_train,axis=0)
#stddev = np.std(x_train,axis=0)
## Preventing zero division
#stddev[stddev<1e-3] = 1
#x_train = (x_train - means)/stddev
#x_val = (x_val - means)/stddev
#x_test = (x_test - means)/stddev
#
## input shape
#act_shape = x_train[0].shape
#num_train = x_train.shape[0]
#
## Creating full input vectors 
#X_train = np.reshape(x_train,(num_train,)+act_shape)
#X_val = np.reshape(x_val,(num_val,)+act_shape)
#X_test = np.reshape(x_test,(num_test,)+act_shape)

# convert class vectors to binary class matrices
# y_train = np.reshape(y_train,(num_train,))
# y_val = np.reshape(y_val,(num_val,))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# def classifyReport(y_true,y_pred):
#   with sess.as_default():
#     y_true = np.asarray(y_true.eval())
#     y_pred = np.asarray(y_pred.eval())
#     return classification_report(y_true,y_pred)

# FC classifier network  trained on activations
model_class = Sequential()
model_class.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu',init="he_normal"))
model_class.add(BatchNormalization())
model_class.add(Dropout(0.5))
model_class.add(Dense(256,activation='relu',init="he_normal"))
model_class.add(BatchNormalization())
model_class.add(Dropout(0.5))
model_class.add(Dense(nb_classes,activation='softmax'))

# Deeper FC classifier network (lesser params) trained on activations
model_class_1024_256_256 = Sequential()
model_class_1024_256_256.add(Dense(1024,input_shape=(X_train.shape[1],),
    activation='relu',init="he_normal"))
model_class_1024_256_256.add(BatchNormalization())
model_class_1024_256_256.add(Dropout(0.6))
model_class_1024_256_256.add(Dense(256,activation='relu',init="he_normal"))
model_class_1024_256_256.add(BatchNormalization())
model_class_1024_256_256.add(Dropout(0.6))
model_class_1024_256_256.add(Dense(256,activation='relu',init="he_normal"))
model_class_1024_256_256.add(BatchNormalization())
model_class_1024_256_256.add(Dropout(0.7))
model_class_1024_256_256.add(Dense(nb_classes,activation='softmax'))

# FC regression network  trained on activations
model_reg = Sequential()
model_reg.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu',init="he_normal"))
model_reg.add(BatchNormalization())
model_reg.add(Dropout(0.5))
model_reg.add(Dense(256,activation='relu'))
model_reg.add(BatchNormalization())
model_reg.add(Dropout(0.5))
model_reg.add(Dense(1))

# FC regression network  trained on activations
model_reg_2048 = Sequential()
model_reg_2048.add(Dense(2048,input_shape=(X_train.shape[1],),activation='relu'))
model_reg_2048.add(BatchNormalization())
model_reg_2048.add(Dropout(0.6))
model_reg_2048.add(Dense(256,activation='relu'))
model_reg_2048.add(BatchNormalization())
model_reg_2048.add(Dropout(0.5))
model_reg_2048.add(Dense(1))

# FC regression network  trained on activations
model_reg_1024d = Sequential()
model_reg_1024d.add(Dense(1024,input_shape=(X_train.shape[1],),activation='relu'))
model_reg_1024d.add(BatchNormalization())
model_reg_1024d.add(Dropout(0.5))
model_reg_1024d.add(Dense(256,activation='relu'))
model_reg_1024d.add(BatchNormalization())
model_reg_1024d.add(Dropout(0.5))
model_reg_1024d.add(Dense(64,activation='relu'))
model_reg_1024d.add(BatchNormalization())
model_reg_1024d.add(Dropout(0.5))
model_reg_1024d.add(Dense(1))

# Fixing some keras bug
keras.backend.get_session().run(tf.global_variables_initializer())

# Picking the optimizer

if optim=='sgd':
  foptim = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
elif optim=='rmsprop':
  foptim = RMSprop(lr=lr,rho=0.9,epsilon=1e-8,decay=decay)

#### Training Classification or regression models #####

print("Training a classifier with NLL loss\n")

# name to save model
modelName = '1024-256-256_'+modelName  
model = model_class_1024_256_256
model.compile(loss='categorical_crossentropy',
              optimizer=foptim,
              metrics=['categorical_accuracy'])

# defining callback functions for saving models etc. Saves model with 
# best validation accuracy
checkPointer = ModelCheckpoint(filepath="./savedModels/"+modelName+'.hdf5',
                               monitor='val_loss',verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-7)
history = model.fit(X_train, Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_data=(X_val, Y_val),
        shuffle=True,callbacks=[checkPointer])

best_model = keras.models.load_model("./savedModels/"+modelName+'.hdf5')
test_prediction = best_model.predict_classes(X_test)
print("\nPrinting results on test dataset for best saved model: \n")
temp = classification_report(y_test,test_prediction)
print(temp)
temp = confusion_matrix(y_test,test_prediction)
print(temp)
print("Test accuracy: %0.2f " % sklearn.metrics.accuracy_score(y_test,test_prediction))

hist = history.history
np.save("./savedModels/"+modelName+'.npy',hist)

plt.plot(hist['categorical_accuracy'])
plt.plot(hist['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("./plots/losscurves/"+modelName+'_acc'+'.png')

# summarize history for loss
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig("./plots/losscurves/"+modelName+'_loss'+'.png')

# Possible Data Augmentation 
# if not data_augmentation:
    # print('Not using data augmentation.')
    # history = model.fit(X_train, Y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           validation_data=(X_val, Y_val),
    #           shuffle=True,callbacks=[checkPointer])
# else:
#     print('Using real-time data augmentation.')

#     # this will do preprocessing and realtime data augmentation
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
    
#     datagen.fit(X_train)
    
#     model.fit_generator(datagen.flow(X_train, Y_train,
#                         batch_size=batch_size),
#                         samples_per_epoch=X_train.shape[0],
#                         nb_epoch=nb_epoch,
#                         validation_data=(X_val, Y_val))

# temp = classification_7eport(y_val,model.predict_classes(X_val))
# print(temp)
# temp = confusion_matrix(y_val,model.predict_classes(X_val))
# print(temp)


