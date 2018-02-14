# cs341-ibm-seti

Classifying signals and simulations representing data from the Allen Telescope Array (ATA), in partnership with Adam Cox (@gadamc) from IBM. 

**Representative (clean) signal types**:
![Signal Classes](https://raw.githubusercontent.com/akashmjn/cs341-ibm-seti/master/report/signal_classes.png)

The main challenge was that a large fraction of samples have a very poor SNR, with many examples hard to discern by eye. 

**SETINet V3 Activations for poor SNR example**:
![SETINetV3 Activations](https://raw.githubusercontent.com/akashmjn/cs341-ibm-seti/master/report/setinetv3_activations.png) 

**Final presentation:** [link](https://docs.google.com/presentation/d/e/2PACX-1vSCqoerGFaEWE1RixF4JAQpdREUC-H57kOQU--OU4yEQY08ZLUpwF4J1ghw0Py-hix5G822xES_h_YX/pub?start=false&loop=false&delayms=3000)

**Final report:** [link](https://akashmjn.github.io/cs341/cs341-seti-final-report.pdf)

## Project structure: 

**commonutils (library)/**
  - datautils: utilities for pre-processing, creating dataset, visualizing activations
  - modelspecs: keras model specifications SETINet V1, V2, V3
  - traceutils: utilities for dynamic-programming based feature extractor (path-trace)
  - nputils, modelutils: misc utility functions for pre-processing, debugging
  
**Akash/**
  - Various scripts / notebooks for training models, documenting results
  
**matlab-exploration/**
  - experimentation with image / signal processing techniques 
