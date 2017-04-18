import os,sys
import numpy as np
import nputils
import matplotlib.pyplot as plt

import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from PIL import Image

def saveImageFromSpec(spec,imsize,save=False,filename=None):
    """
    Saves an image generated from an array containing a spectrogram
    """
    spec = nputils.bin_ndarray(spec[1:,:],(128,2048),operation='average')
    dpi = 96.0
    fig = plt.figure(frameon=False,figsize=(imsize[0]/dpi,imsize[1]/dpi))
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # fig, ax = plt.subplots(figsize=(20, 10))
    # ax.imshow(np.log(spec), aspect = 0.5*float(spec.shape[1]) / spec.shape[0])
    ax.imshow(np.log(spec), aspect = 'auto')
    fig.savefig("{}.jpg".format(filename),dpi=dpi)
    plt.close('all')



def modelActivations(model,layer_name,input,show=False,save=False,poolfit=None):
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
            strides=fstride,input_shape=(None,None,512)))
        output = model_pool.predict(output)

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

    if show==True:
        plt.imshow(stitched_filters)
        plt.show()

    if save==True:
        # save the result to disk
        plt.imsave(layer_name+'_stitched_filters_%dx%d.png' % (m, n), stitched_filters)

    return output

######

def generateAllActivations(dirpath,savedir):

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
            act = modelActivations(base_model,'block5_pool',imgBatch)
            for k in range(nBatchSize):
                np.save( os.path.join(savedir,fileBatch[k].split(".")[0]+'.npy') , act[k,:,:,:] )
            img = np.asarray(Image.open( os.path.join(dirpath,imgfile) )).astype('float32')
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img) 
            imgBatch[j] = img 
            fileBatch[j] = imgfile

