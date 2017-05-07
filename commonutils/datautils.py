import os,sys
import numpy as np
import nputils
import matplotlib.pyplot as plt
import collections
import pandas as pd
import h5py

import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import np_utils

from PIL import Image

def saveImageFromSpec(spec,imsize,binFactor,save=False,filename=None):
    """
    Saves an image generated from an array containing a spectrogram
    """
    specShape = spec.shape
    spec = nputils.bin_ndarray(spec[1:,:],(specShape[0]-1,specShape[1]/binFactor),
            operation='average')
    dpi = 96.0
    fig = plt.figure(frameon=False,figsize=(imsize[0]/dpi,imsize[1]/dpi))
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.imshow(np.log(spec), aspect = 0.5*float(spec.shape[1]) / spec.shape[0])
    plt.set_cmap('jet')
    ax.imshow(np.log(spec), aspect = 'auto')
    fig.savefig("{}.jpg".format(filename),dpi=dpi)
    plt.close('all')



def modelActivations(model,layer_name,input,poolfit=None,show=False,save=False,):
    """
    Generates activations for a kxmxnx3 array pre-processed image
    @param poolfit Tuple if size 2 indicating target size for each activation
    """

    model_selection = Model(input=model.input, output=model.get_layer(layer_name).output)

    if (len(input.shape) != 4 or input.shape[3] != 3 or 
        input.dtype not in ['float32','float64']):
        raise IndexError("Input should be float array of kxmxnx3: \n")
    output = model_selection.predict(input)

    # Running a coarse pooling to fit activations to the target size 
    if poolfit:
        model_pool = keras.models.Sequential()
        fsize = (np.ceil(output.shape[1]*1.0/poolfit[0]).astype(int),
                np.ceil(output.shape[2]*1.0/poolfit[1]).astype(int))
        fstride = (np.floor(output.shape[1]*1.0/poolfit[0]).astype(int),
                np.floor(output.shape[2]*1.0/poolfit[1]).astype(int))
        print output.shape
        print fsize
        print fstride
        model_pool.add(keras.layers.MaxPooling2D(pool_size=fsize,
            strides=fstride,input_shape=(None,None,output.shape[3])))
        output = model_pool.predict(output)

    # Creating an image to save activation maps
    if show or save:
        (img_width,img_height) = (output.shape[1],output.shape[2])
        nactivation = output.shape[3]
    
        if np.sqrt(nactivation) % 1 == 0:
            m = int(np.sqrt(nactivation))
            n = m
        elif np.sqrt(nactivation/2) % 1 == 0:
            if img_width > img_height:
                m = int(np.sqrt(nactivation/2))
                n = int(nactivation/m)
            else:
                n = int(np.sqrt(nactivation/2))
                m = int(nactivation/n)            
        else:
            raise IndexError('Not figured out how to layout %d activations yet' % nactivation)
    
        # build a picture with enough space for
        # m x n filters with a 1px margin in between
        margin = 1
        width = m * img_width + (m - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height))
    
        # fill the picture with our saved filters
        for i in range(m):
            for j in range(n):
                img = output[0][:,:,i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height] = img
        plt.set_cmap('jet')
        plt.imshow(stitched_filters)

    if show==True:
        plt.show()

    if save==True:
        # save the result to disk
        plt.imsave(layer_name+'_stitched_filters_%dx%d.png' % (m, n), stitched_filters)

    return output

######

def generateAllActivations(dirpath,savedir,layer_name,poolfit=None):
    """
    Reads in all images from a folder and generates activations for them in batches
    As it is a long process that will likely be interrupted, it checks and only processes
    files that have not been generated so far. 
    """

    os.system('mkdir -p '+savedir)
    # Listing out image files in source directory
    imagefiles = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    ext = imagefiles[0].split('.')[1]
    # imageIDs = [int(f.split('.')[0]) for f in imagefiles]
    generatedFiles = [f.split('.')[0]+'.'+ext for f in os.listdir(savedir) if 
            os.path.isfile(os.path.join(savedir, f))]
    # Filter by files that have already been generated
    imagefiles = [f for f in imagefiles if f not in generatedFiles]
    nFiles = len(imagefiles)


    ## Actually iterating through files and creating batches to generate activations
    nBatchSize = 8
    imtest = np.asarray(Image.open( os.path.join(dirpath,imagefiles[0]) )).astype('float32')
    # Making batches to run in bulk on GPU
    (imWidth,imHeight) = imtest.shape[0:2]
    imgBatch = np.zeros((nBatchSize,imWidth,imHeight,3))
    base_model = VGG16(weights='imagenet', include_top=False)
    # Creating batches to process on the GPU
    fileBatch = {}
    j = 0
    for i in range(nFiles):
        j = i % nBatchSize
        imgfile = imagefiles[i]
        print('\rFile %d / %d: '+imgfile) % (i,nFiles),
        # for non multiple i (or the beginning) fill up the batches
        if j!= 0 or i==0:
            img = np.asarray(Image.open( os.path.join(dirpath,imgfile) )).astype('float32')
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img) 
            imgBatch[j] = img 
            fileBatch[j] = imgfile
        # for multiple i (or the end) process the batch and restart
        if (j == 0 and i > 0) or i==(nFiles-1):
            act = modelActivations(base_model,layer_name,imgBatch,poolfit)
            for k in range(nBatchSize):
                np.save( os.path.join(savedir,fileBatch[k].split(".")[0]+'.npy') , act[k,:,:,:] )
            img = np.asarray(Image.open( os.path.join(dirpath,imgfile) )).astype('float32')
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img) 
            imgBatch[j] = img 
            fileBatch[j] = imgfile


def _createDatasetHelper(x_array,y_array,id_array,LabelDict,fileList,basePath,loaderFn,prompt):
    nFiles = len(fileList)
    for i in range(nFiles):
        filename = fileList[i]
        print('\r{} {} / {}'.format(prompt,i+1,nFiles)),
        dataPoint = loaderFn(os.path.join(basePath,filename))
        x_array[i,:] = dataPoint
        fileID = filename.split(".")[0]
        id_array.append(fileID)
        y_array[i] = LabelDict[fileID]  
    print('\n')

### Functions to load in and combine all model activations into datasets

def createDataset(sourcePath,fileListPath,nval,ntest,destFilename,loadImages,ntrain=None):
    # Loading in list of files
    files = [f for f in os.listdir(sourcePath)]
    ext = list(set([f.split('.')[1] for f in files]))[0]
    print 'Picking files of extension %s' % ext
    files = [f for f in files if f.endswith(ext)]
    
    # If ntrain specified, this is used to create datasets of smaller
    # size by randomly picking files from the entire file list
    if ntrain:
        nFull = min(len(files),ntrain+ntest+nval)
        np.random.seed(2)
        sampleIndices = np.random.permutation(range(len(files)))[0:nFull]
        files = [ files[i] for i in sampleIndices ]
    else:
        ntrain = len(files) - (ntest+nval)
    n = len(files)
    
    # Splitting training and test data
    print 'The number of files is %d' %(n)    
    # ntrain = n - int(n*traintestSplit)
    # ntest  = int(n*traintestSplit)

    # Reading in all the data labels into a dict for use below
    # key : file_index (000100 etc.) 
    Label_dict = collections.defaultdict(list)
    fileListDF = pd.read_csv("fileList.csv",dtype={'file_index':str})
    for i in range(len(fileListDF.index)):
        Label_dict[fileListDF.ix[i]['file_index']]=fileListDF.ix[i]['label']

    if loadImages: 
        size = image.img_to_array(image.load_img(os.path.join(sourcePath,files[0]))).shape
        loaderFn = lambda x: image.img_to_array(image.load_img(x))
    else:
        size = np.load(os.path.join(sourcePath,files[0])).flatten().shape
        loaderFn = lambda x: np.load(x).flatten()

    '''
    Store the input files as a x_train, y_train, x_val and y_val ,x_test and y_test.
    '''
    x_train = np.zeros((ntrain,)+size)
    y_train = np.zeros((ntrain,))
    train_ids = []
    _createDatasetHelper(x_train,y_train,train_ids,
            Label_dict,files[0:ntrain],sourcePath,loaderFn,'Training set')
    
    x_val = np.zeros((nval,)+size)
    y_val = np.zeros((nval,))
    val_ids = []
    _createDatasetHelper(x_val,y_val,val_ids,
            Label_dict,files[ntrain:ntrain+nval],sourcePath,loaderFn,'Validation set')

    x_test = np.zeros((ntest,)+size)
    y_test = np.zeros((ntest,))
    test_ids = []
    _createDatasetHelper(x_test,y_test,test_ids,
            Label_dict,files[ntrain+nval:ntrain+nval+ntest],sourcePath,loaderFn,'Test set')

    # distribution of labels in train / test
    count_dict_train = {}
    count_dict_val = {}
    count_dict_test = {}
    for i in range(int(np.max(y_train))+1):
        count_dict_train[i] = np.count_nonzero(y_train==i)
        count_dict_val[i] = np.count_nonzero(y_val==i)
        count_dict_test[i] = np.count_nonzero(y_test==i)

    # Should probably change this part 
    print 'Dim of data: %d' % x_train[0,:].shape[0]
    print 'Number of training images = %d' %(ntrain)
    print 'Number of validation images = %d' %(nval)
    print 'Number of test images = %d' %(ntest)

    print 'Distribution in training images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_train[0],count_dict_train[1],count_dict_train[2],
            count_dict_train[3],count_dict_train[4])
    print 'Distribution in validation images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_val[0],count_dict_val[1],count_dict_val[2],count_dict_val[3],count_dict_val[4])
    print 'Distribution in test images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_test[0],count_dict_test[1],count_dict_test[2],count_dict_test[3],count_dict_test[4])

    # Creating a compiled dataset, and then saving it to file
    dataset = {'x_train':x_train,'y_train':y_train,'train_ids':train_ids,
            'x_val':x_val,'y_val':y_val,'val_ids':val_ids,
            'x_test':x_test,'y_test':y_test,'test_ids':test_ids}

    with h5py.File(destFilename,'w') as hf:
        for key in dataset.keys():
            hf.create_dataset(key,data=dataset[key])


# Deprecated function
def createActivationsDataset(actPath,fileListPath,nval,ntest,actFilename,ntrain=None):
    print("WARNING: This function has changed to createDataset - see commonutils for usage")
    createDataset(actPath,fileListPath,nval,ntest,actFilename,loadImages=False,ntrain=None)


def loadDataset(hdf5filepath,scale=True,ntrain=None,fixSkew=None,subsetClasses=None):
    """
    subsetClasses: dict mapping a subset of labels to the required labels
    Ignore fixSkew for now
    """

    with h5py.File(hdf5filepath,'r') as hf:
        x_train = np.array(hf.get('x_train'))
        y_train = np.array(hf.get('y_train'))
        train_ids = np.array(hf.get('train_ids'))
        x_val = np.array(hf.get('x_val'))
        y_val = np.array(hf.get('y_val'))
        val_ids = np.array(hf.get('val_ids'))
        x_test = np.array(hf.get('x_test'))
        y_test = np.array(hf.get('y_test'))
        test_ids = np.array(hf.get('test_ids'))
    
    # If argument passed, subset the data
    if subsetClasses:
        n = y_train.shape[0]
        indices = np.array([y_train[i] in subsetClasses.keys() for i in range(n)])
        x_train = x_train[indices]
        train_ids = train_ids[indices]
        y_train = np.array([subsetClasses[label] for label in y_train[indices]])
        n = y_val.shape[0]
        indices = np.array([y_val[i] in subsetClasses.keys() for i in range(n)])
        x_val = x_val[indices]
        val_ids = val_ids[indices]
        y_val = np.array([subsetClasses[label] for label in y_val[indices]])
        n = y_test.shape[0]
        indices = np.array([y_test[i] in subsetClasses.keys() for i in range(n)])
        x_test = x_test[indices]
        test_ids = test_ids[indices]
        y_test = np.array([subsetClasses[label] for label in y_test[indices]])

    # If specified resample training data
    if ntrain:
        np.random.seed(2)
        indices = np.random.permutation(range( x_train.shape[0] ))[0:ntrain]
        x_train = x_train[indices]
        y_train = y_train[indices]
        train_ids = train_ids[indices]

    # distribution of labels in train / test
    # TODO: Need to generalize this for this case
    count_dict_train = {}
    count_dict_val = {}
    count_dict_test = {}
    for i in range(5):
        count_dict_train[i] = np.count_nonzero(y_train==i)
        count_dict_val[i] = np.count_nonzero(y_val==i)
        count_dict_test[i] = np.count_nonzero(y_test==i)

    # Resampling data 
    # TODO: Need to generalize this code 
    if fixSkew:
        train_dist = {}
        train_resample = {}
        bdist = 0.2
        for k,v in count_dict_train.items():
            train_dist[k] = v*1.0/ntrain
            if (bdist-train_dist[k])>0.01:
                train_resample[k] = int(bdist*ntrain) - v

        for k,v in train_resample.items():
            if train_dist[k]!=0:
                x_train_label_k = x_train[y_train==k]
                np.random.seed(2)
                resampled = np.random.choice(x_train_label_k.shape[0],v)
                x_train = np.append(x_train,x_train[resampled],axis=0)
                y_train = np.append(y_train,np.repeat(k,len(resampled)),axis=0)
                count_dict_train[k] = int(bdist*ntrain)
    
    # Centering and scaling data
    num_val = x_val.shape[0]
    num_test = x_test.shape[0]

    if scale:
        means = np.mean(x_train,axis=0)
        stddev = np.std(x_train,axis=0)
        # Preventing zero division
        stddev[stddev<1e-3] = 1
        x_train -= means
        x_train /= stddev
        x_val -= means
        x_val /= stddev
        x_test -= means
        x_test /= stddev

    # input shape
    act_shape = x_train[0].shape
    num_train = x_train.shape[0]

    # Creating full input vectors 
    x_train = np.reshape(x_train,(num_train,)+act_shape)
    x_val = np.reshape(x_val,(num_val,)+act_shape)
    x_test = np.reshape(x_test,(num_test,)+act_shape)

    # convert class vectors to binary class matrices
    # y_train = np.reshape(y_train,(num_train,))
    # y_val = np.reshape(y_val,(num_val,))
#    y_train = np_utils.to_categorical(y_train, nb_classes)
#    y_val = np_utils.to_categorical(y_val, nb_classes)
#    y_test = np_utils.to_categorical(y_test, nb_classes)


    print 'Dim of data: %d' % x_train[0,:].shape[0]

    print 'Number of training images = %d' %(y_train.shape[0])
    print 'Number of validation images = %d' %(y_val.shape[0])
    print 'Number of test images = %d' %(y_test.shape[0])

    print 'Distribution in training images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_train[0],count_dict_train[1],count_dict_train[2],
            count_dict_train[3],count_dict_train[4])
    print 'Distribution in validation images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_val[0],count_dict_val[1],count_dict_val[2],count_dict_val[3],count_dict_val[4])
    print 'Distribution in test images: \n0 - %d \n1 - %d \n2 - %d \n3 - %d \n4 - %d'%(\
            count_dict_test[0],count_dict_test[1],count_dict_test[2],count_dict_test[3],count_dict_test[4])

    return  {'x_train':x_train,'y_train':y_train,'train_ids':train_ids,
            'x_val':x_val,'y_val':y_val,'val_ids':val_ids,
            'x_test':x_test,'y_test':y_test,'test_ids':test_ids}               

