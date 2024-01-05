import codecs
import math
import numpy as np
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras import backend as K
import os
from random import shuffle
seed = 1
np.random.seed(seed)
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.val_acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.losses = {'batch': [], 'epoch': []}

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def coefficient_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (SS_res / (SS_tot + K.epsilon()))

def get_dataset(data_file, label_file, vec_num, test_size=0.2):
    dataFile1 = data_file
    dataFile2 = label_file
    data = scio.loadmat(dataFile1)
    label = scio.loadmat(dataFile2)
    X = np.transpose(data['data'])
    Y = np.transpose(label['label'])
    # Generate the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=0)
    X_train = X_train.reshape(np.size(X_train, 0), vec_num, 1)
    X_val = X_val.reshape(np.size(X_val, 0), vec_num, 1)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    return X_train, X_val, Y_train, Y_val
