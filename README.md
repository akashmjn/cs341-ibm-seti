# cs341-ibm-seti

Classifying signals and simulations representing data from the Allen Telescope Array (ATA), in partnership with Adam Cox (@gadamc) from IBM. 

**Final presentation:** [link](https://docs.google.com/presentation/d/e/2PACX-1vSCqoerGFaEWE1RixF4JAQpdREUC-H57kOQU--OU4yEQY08ZLUpwF4J1ghw0Py-hix5G822xES_h_YX/pub?start=false&loop=false&delayms=3000)

**Final report:** [link](https://github.com/akashmjn/cs341-ibm-seti/blob/master/report/cs341-seti-final-report.pdf)

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
