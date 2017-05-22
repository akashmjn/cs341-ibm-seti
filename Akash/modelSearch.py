import os,sys

datasetPath = "../data/imagesDataset_512x256_8/"
augmentFactor = 2
nEpochs = 50
optim = 'adam'
dropout = 0.6
lrAnneal = 0.07

lrList = [2e-5,2e-5,3e-5,5e-5]
lrAnnealList = [1e-7,0.07,0.07,0.07]
decay = 1e-7

for i in range(len(lrList)):
    lr = lrList[i]
    lrAnneal = lrAnnealList[i]
    print "\n\n ##### Running a new run with ###### "
    print '{} {} {} {} {} {} {}\n'.format(datasetPath,
        augmentFactor,nEpochs,optim,lr,lrAnneal,dropout)
    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {}'.format(datasetPath,
        augmentFactor,nEpochs,optim,lr,decay,dropout,lrAnneal))

