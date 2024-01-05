# Overview

This repo contains demonstrations of using the two-dimensional convolutional neural network (2D-CNN), 1D-CNN and regression model to identify the Raman spectra of different pH values. The Raman spectral dataset here contains the training set, validation set and test set randomly divided from 14137 spectra from 41 SERS chips mentioned in the paper.


## Requirements

The code in this repo has been tested with the following software versions:
- Python 3.7.16
- Tensorflow 2.7.0
- Keras 2.7.0
- Scikit-Learn 1.0.2
- Numpy 1.21.6
- Scipy 1.7.3
- Matplotlib 3.5.2


## How to use the package

1. Run the file "train.py" to train and evaluate the 2D-CNN model.
1. Run the file "deepmodel_train.py" in "compare_test" to train and evaluate the 1D-CNN model.
1. Run the file "LinearReg_train.py" in "compare_test" to train and evaluate the traditional regression model.
