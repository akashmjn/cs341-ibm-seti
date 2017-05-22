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

batch_size = 32
datasetPath = args[1]
augmentFactor = int(args[2])
nb_epoch = int(args[3])
optim = args[4]
lr = float(args[5])
decay = float(args[6])
dropout = float(args[7])

# Setting directory paths
trainDataPath = os.path.join(datasetPath,'train')
valDataPath = os.path.join(datasetPath,'validation')
testDataPath = os.path.join(datasetPath,'test')
# Classes to use
nb_classes = 4
classList = ['0-noise','2-narrowband','3-narrowbanddrd','5-squiggle']

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

### Loading in validation data as array 
#subsetClasses = {0.0:0.0,2.0:1.0,3.0:2.0,5.0:3.0}
#dataset = cu.datautils.loadDataset(os.path.join(datasetPath,'imagesDataset_512x256_8.h5'),
#                                        subsetClasses=subsetClasses) # Hardcoded, should change
#X_val = dataset['x_val']
#Y_val = dataset['y_val']
#Y_val = np_utils.to_categorical(Y_val, nb_classes)
#val_ids = dataset['val_ids']
#dataset = None   # free up memory

# Initializing augmentation objects
## Have marked out for sanitycheck. Change after!
train_generator = test_datagen.flow_from_directory(directory=trainDataPath,batch_size=batch_size,
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
modelName = '{}class_{}augment{}dropout{}lr{}decay{}'.format(nb_classes,optim,
        augmentFactor,dropout,lr,decay)
modelName = 'setiNet_256x512_'+modelName  
# model = model_specs.fc_1024_256_256.build(X_train.shape[1],nb_classes)
model = model_specs.setiNet.build((256,512,1),nb_classes,dropout=dropout)
# Fixing some keras bug
keras.backend.get_session().run(tf.global_variables_initializer())
model.compile(loss='categorical_crossentropy',
              optimizer=foptim,
              metrics=['categorical_accuracy'])

#### Definining a bunch of callbacks monitoring/lr scheduling etc. ####

# Saves model with best validation loss
checkPointer = ModelCheckpoint(filepath="../savedModels/"+modelName+'.hdf5',
                               monitor='val_loss',verbose=1, save_best_only=True)
# learning rate scheduling
def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("Learning rate: %0.2f",lrate)
    return lrate
lrate = LearningRateScheduler(step_decay)
# reducing learning rate if performance stalls
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-7)
# early stopping?
earlyStop = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=5)
# tensorboard
tensorboard = TensorBoard(log_dir='./tensorboardLogs/'+modelName,histogram_freq=2,write_images=True)
                

#### Training Classification or regression models #####

print("Training a classifier with NLL loss\n")

#history = model.fit_generator(train_generator,batch_size=batch_size,
#        steps_per_epoch = train_generator.n//batch_size*augmentFactor,
#        epochs=nb_epoch,validation_data=(X_val,Y_val),
#        callbacks=[checkPointer,lrate,tensorboard])

history = model.fit_generator(train_generator,
        steps_per_epoch = train_generator.n//batch_size*augmentFactor,
        epochs=nb_epoch,validation_data=validation_generator,
        validation_steps=validation_generator.n//batch_size,
        callbacks=[checkPointer,lrate,tensorboard])

#### TO DO: Complete modifying this ####

#best_model = keras.models.load_model("./savedModels/"+modelName+'.hdf5')
#test_prediction = np.argmax(best_model.predict(X_test),axis=1)
#print("\nPrinting results on test dataset for best saved model: \n")
#temp = classification_report(y_test,test_prediction)
#print(temp)
#temp = confusion_matrix(y_test,test_prediction)
#print(temp)
#print("Test accuracy: %0.2f " % sklearn.metrics.accuracy_score(y_test,test_prediction))
#
#hist = history.history
#np.save("./savedModels/"+modelName+'.npy',hist)
#
#plt.plot(hist['categorical_accuracy'])
#plt.plot(hist['val_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("./plots/losscurves/"+modelName+'_acc'+'.png')
#
## summarize history for loss
#plt.plot(hist['loss'])
#plt.plot(hist['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper right')
#plt.savefig("./plots/losscurves/"+modelName+'_loss'+'.png')


