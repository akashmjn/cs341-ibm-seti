# Training a CNN on images
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
K.set_image_dim_ordering('tf')

import os,sys
sys.path.append('/home/cs341seti/cs341-ibm-seti/')
import matplotlib.pyplot as plt
#import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import re
import collections
import sklearn, math
import commonutils as cu 
import model_specs
from sklearn import svm
from sklearn.externals import joblib

# Taking in command line args
# python cnnModelTrain.py DATASETPATH AUGMENTFACTOR nEPOCHS OPTIM LR DECAY DROPOUT 
args = sys.argv
print (args)

batch_size = 32
datasetPath = args[1]
run_name = args[2]
print(args[3])
augmentFactor = int(args[3])
nb_epoch = int(args[4])
epoch_offset = int(args[5])
optim = args[6]
lr = float(args[7])
decay = float(args[8])
dropout = float(args[9])
lrAnneal = float(args[10])
kernel_init = args[11]

# Setting directory paths
trainDataPath = os.path.join(datasetPath,'train')
valDataPath = os.path.join(datasetPath,'validation')
testDataPath = os.path.join(datasetPath,'test')
tbpath = './tblogs-{}/'.format(run_name)
modelpath = '../savedModels/cnnModels-{}/'.format(run_name)
os.system('mkdir -p {}'.format(modelpath))
weightspath = "../savedModels/cnnModels-2class-aug-fullsearch/setiNetv2_256x512_2class_adamaugment2dropout0.47lr2.76e-04anneal0.11.hdf5"
# Classes to use
nb_classes = 2
classList = ['0-noise','7-signal-basic']

#### Preparing data #### 
# Creating augmentation objects
train_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,
                                   rotation_range=0, width_shift_range=0.2,
                                   height_shift_range=0.1,zoom_range=0.1,
                                   horizontal_flip=True, vertical_flip = True, fill_mode='constant')
test_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)

# Load in a sample dataset to fix normalization / rescaling params
trainImgDir = os.path.join(datasetPath,'train_one_folder')
files = [f for f in os.listdir(trainImgDir) if f.endswith('.jpg')]
size = img_to_array(load_img(os.path.join(trainImgDir,files[0]),grayscale=True)).shape
trainDataArray = np.zeros((len(files)//5,)+size) # Only 1/5 the data, because memory constraints
for i in range(len(files)//5):
    trainDataArray[i] = img_to_array(load_img(os.path.join(trainImgDir,files[i]),grayscale=True))
# Initializing data standardization
train_datagen.fit(trainDataArray)
test_datagen.fit(trainDataArray)

# Initializing augmentation objects
train_generator = train_datagen.flow_from_directory(directory=trainDataPath,batch_size=batch_size,
        class_mode='categorical',classes=classList,target_size=(256,512),color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(directory=valDataPath,batch_size=batch_size,
        class_mode='categorical',classes=classList,target_size=(256,512),color_mode='grayscale')
       
#### Preparing model ####

# Picking the optimizer
if optim=='sgd':
  foptim = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
elif optim=='rmsprop':
  foptim = RMSprop(lr=lr,rho=0.9,epsilon=1e-8,decay=decay)
elif optim=='adam':
  foptim = Adam(lr=lr,decay=decay)

# name to save model
modelName = '{}class_{}offset{}augment{}dropout{:.2f}lr{:.2e}anneal{:.2f}'.format(nb_classes,optim,
        epoch_offset,augmentFactor,dropout,lr,lrAnneal)
modelName = 'setiNetv2_256x512_'+modelName  
# model = model_specs.fc_1024_256_256.build(X_train.shape[1],nb_classes)
model = model_specs.setiNet_v2.build((256,512,1),nb_classes,dropout=dropout,init=kernel_init,
        weightsPath=None)
## Fixing some keras bug
#keras.backend.get_session().run(tf.global_variables_initializer())
#model.compile(loss='categorical_crossentropy',
#              optimizer=foptim,
#              metrics=['categorical_accuracy'])

#### Definining a bunch of callbacks monitoring/lr scheduling etc. ####

# Saves model with best validation loss
checkPointer = ModelCheckpoint(filepath=modelpath+modelName+'.hdf5',
                               monitor='val_loss',verbose=1, save_best_only=True)
# learning rate scheduling
def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("Learning rate: %0.2f",lrate)
    return lrate

def anneal_decay(epoch):
    initial_lrate = lr
    beta = lrAnneal
    lrate = initial_lrate / (1 + beta*(epoch+epoch_offset))
    print("Learning rate: ",lrate)
    return lrate

#lrate = LearningRateScheduler(step_decay)
lrate = LearningRateScheduler(anneal_decay)
# reducing learning rate if performance stalls
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-7)
# early stopping?
earlyStop = EarlyStopping(monitor='val_loss',min_delta=0.01,patience=10)
# tensorboard
tensorboard = TensorBoard(log_dir=tbpath+modelName,histogram_freq=2,write_images=True)
                

#### Training Classification or regression models #####

print("Training a classifier with NLL loss\n")

#history = model.fit_generator(train_generator,batch_size=batch_size,
#        steps_per_epoch = train_generator.n//batch_size*augmentFactor,
#        epochs=nb_epoch,validation_data=(X_val,Y_val),
#        callbacks=[checkPointer,lrate,tensorboard])

history = model.fit_generator(train_generator,class_weight={0:4.0,1:1.0},
        steps_per_epoch = train_generator.n//batch_size*augmentFactor,
        epochs=nb_epoch,validation_data=validation_generator,
        validation_steps=validation_generator.n//batch_size,
        callbacks=[checkPointer,lrate,tensorboard])

