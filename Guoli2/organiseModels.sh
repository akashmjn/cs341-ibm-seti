BASEPATH="../savedModels/"
BASEFOLDER=$( echo $1 | sed 's/tblogs-/cnnModels-/g')
mkdir -p $BASEPATH$BASEFOLDER
for MODELNAME in $(ls $1)
do
    #MODELFILE=$(find ../savedModels -name $MODELNAME"*")  
    MODELFILE=$MODELNAME.hdf5
    echo Moving model : $BASEPATH$MODELFILE
    mv $BASEPATH$MODELFILE $BASEPATH$BASEFOLDER
done

