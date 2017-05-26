import os,sys
import numpy as np

datasetPath = "../data/imagesDataset_512x256_8/"
augmentFactor = 2
nEpochs = 50
optim = 'adam'
dropout = 0.6
lrAnneal = 0.07
kernel_init = 'he_uniform'
decay = 1e-7

### Params for organised search
#lrList = [2e-5,2e-5,3e-5,5e-5]
#lrAnnealList = [1e-7,0.07,0.07,0.07]
#
### Organised search
#for i in range(len(lrList)):
#    lr = lrList[i]
#    lrAnneal = lrAnnealList[i]
#    print "\n\n ##### Running a new run with ###### "
#    print '{} {} {} {} {} {} {} {}\n'.format(datasetPath,
#        augmentFactor,nEpochs,optim,lr,lrAnneal,dropout,kernel_init)
#    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {} {}'.format(datasetPath,
#        augmentFactor,nEpochs,optim,lr,decay,dropout,lrAnneal,kernel_init))

## Params for random search 
lrLims = (-5,-4)
annealLims = (0.05,0.2)
dropoutLims = (0.3,0.6)

## Random search
for i in range(10):
    lr = 10**np.random.uniform(lrLims[0],lrLims[1])
    print(lr)
    lrAnneal = np.random.uniform(annealLims[0],annealLims[1])
    dropout = np.random.uniform(dropoutLims[0],dropoutLims[1])
    print "\n\n ##### Running a new run with ###### "
    print '{} {} {} {} {:.2e} {:.2f} {:.2f} {}\n'.format(datasetPath,
        augmentFactor,nEpochs,optim,lr,lrAnneal,dropout,kernel_init)
    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {} {}'.format(datasetPath,
        augmentFactor,nEpochs,optim,lr,decay,dropout,lrAnneal,kernel_init))

