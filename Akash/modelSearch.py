import os,sys
import numpy as np

datasetPath = "../data/imagesDataset_512x256_8/"
run_name = "2class-selected-models"
augmentFactor = 5
nEpochs = 50
optim = 'adam'
dropout = 0.47
lrAnneal = 0.11
kernel_init = 'he_uniform'
decay = 1e-7

## Params for organised search
#lrList = [5.57e-5,5.57e-5]
lrList = [2.76e-4,2.76e-4]
#lrAnnealList = [0.11,0.05]
epoch_offsetList = [40,50]

## Organised search
for i in range(len(lrList)):
    lr = lrList[i]
    epoch_offset = epoch_offsetList[i]
    #lrAnneal = lrAnnealList[i]
    print "\n\n ##### Running a new run with ###### "
    print '{} {} {} {} {} {} {:.2e} {:.2f} {:.2f} {}\n'.format(datasetPath,run_name,
        augmentFactor,nEpochs,epoch_offset,optim,lr,lrAnneal,dropout,kernel_init)
    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {} {} {} {}'.format(datasetPath,run_name,
        augmentFactor,nEpochs,epoch_offset,optim,lr,decay,dropout,lrAnneal,kernel_init))
#    print "\n\n ##### Running a new run with ###### "
#    print '{} {} {} {} {} {} {} {}\n'.format(datasetPath,
#        augmentFactor,nEpochs,optim,lr,lrAnneal,dropout,kernel_init)
#    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {} {}'.format(datasetPath,
#        augmentFactor,nEpochs,optim,lr,decay,dropout,lrAnneal,kernel_init))

### Params for random search 
#lrLims = (-4.5,-3.5)
#annealLims = (0.05,0.2)
#dropoutLims = (0.2,0.6)
#
### Random search
#for i in range(10):
#    lr = 10**np.random.uniform(lrLims[0],lrLims[1])
#    print(lr)
#    lrAnneal = np.random.uniform(annealLims[0],annealLims[1])
#    dropout = np.random.uniform(dropoutLims[0],dropoutLims[1])
#    print "\n\n ##### Running a new run with ###### "
#    print '{} {} {} {} {} {:.2e} {:.2f} {:.2f} {}\n'.format(datasetPath,run_name,
#        augmentFactor,nEpochs,optim,lr,lrAnneal,dropout,kernel_init)
#    os.system('python cnnModelTrain.py {} {} {} {} {} {} {} {} {} {}'.format(datasetPath,run_name,
#        augmentFactor,nEpochs,optim,lr,decay,dropout,lrAnneal,kernel_init))
#
