# Overview

This repo contains demonstrations of using the 2D-CNN to predict pH and PSA values of the Raman spectra. The Raman spectral dataset here is the demo dataset, containing the training set, validation set and test set randomly divided from 350 spectra.


## Requirements

The code in this repo has been trained and tested on a NVIDIA GeForce RTX 3090 GPU with the following software versions:
- Python 3.8.20
- Pytorch 2.4.1+cu118
- Numpy 1.24.3
  

## How to use the package

Run the file "Train.py" to train and test the 2D-CNN.
1.Parameter settings:  
(1) train_p: False/True, which means whether to train the model or not.  
(2) predict_p: False/True, which means whether to test the model or not.  
(3) base_corr: False/True, which means whether to apply baseline correction to spectra.  
(4) show_result: False/True, which means whether to show the predicted outputs.  
2.Output explanation：  
Output contains:  
(1)MAE,SD,SSE of each pH value and all of pH values.  
(2)MAE,SD,SSE of each PSA value and all of PSA values.  

The training time using our demo dataset and model is about 20 minutes. 
