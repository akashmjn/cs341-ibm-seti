import os,sys

datasetPath = "../data/imagesDataset_512x256_8/"
augmentFactor = 2
nEpochs = 20
optim = 'rmsprop'

lrList = [1e-5,2e-5,3e-5]
decay = 1e-7

for lr in lrList:
    print "\n\n ##### Running a new run with ###### \n"
    os.system('python cnnModelTrain.py {} {} {} {} {} {}'.format(datasetPath,
        augmentFactor,nEpochs,optim,lr,decay))

