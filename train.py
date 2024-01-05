"""
Train and evaluate the Raman model.
"""
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
import numpy as np
from init.prepare import get_dataset, LossHistory,get_dataset_ph
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from datetime import datetime
from models.resnet import resnet_34
import scipy.io as scio
from init.prepare import coefficient_determination
from tensorflow.keras.optimizers import Nadam, Adam
np.set_printoptions(threshold=np.inf)
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Initialization
batch_size = 8

checkpoint_save_path = os.path.abspath(os.path.dirname(__file__)) + '/check_point/ours.ckpt'

epochs = 200
initial_lr = 1e-3


early_stopping = EarlyStopping(monitor='val_loss',
                               patience=15,
                               verbose=2)

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=10,
                               verbose=2,
                               mode='min',
                               cooldown=0,
                               min_lr=1e-7)

model_checkpoint = ModelCheckpoint(checkpoint_save_path,
                                   monitor='val_loss',
                                   save_weights_only=True,
                                   save_best_only=True,
                                   mode='min')

vec=287

data_path1 = os.path.abspath(os.path.dirname(__file__))+'/dataset/10361_airpls_divide_peak1.mat'
label_path1 = os.path.abspath(os.path.dirname(__file__))+'/dataset/10361_ph_label.mat'
data_path = os.path.abspath(os.path.dirname(__file__))+'/dataset/solo_airpls_divide_peak1.mat'
label_path = os.path.abspath(os.path.dirname(__file__))+'/dataset/solo_ph_label.mat'

X_train, X_val, Y_train, Y_val = get_dataset(data_path, label_path, vec_num=vec,test_size=0.2)
X_train1, X_val1, Y_train1, Y_val1 = get_dataset(data_path1, label_path1, vec_num=vec,test_size=0.2)

X_train = np.concatenate([X_train,X_train1],0)
X_val = np.concatenate([X_val,X_val1],0)
Y_train=np.concatenate([Y_train,Y_train1],0)
Y_val=np.concatenate([Y_val,Y_val1],0)

X_train, X_valk, Y_train, Y_valk = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)


# Train
model = resnet_34(vec,[3,4,6,3])
model.build((8, vec, 1))
model.summary()

begin = datetime.now()
model.compile(loss='mse',
                  optimizer=Nadam(learning_rate=1e-3),
                  metrics=[coefficient_determination])
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=1,
          validation_data=(X_valk, Y_valk),
          callbacks=[early_stopping, lr_reducer, model_checkpoint])
end = datetime.now()
print('Training time: ', end-begin)


# Evaluation
my_model = resnet_34(vec,[3,4,6,3])
my_model.load_weights(checkpoint_save_path)
my_model.compile(loss='mse',
                  optimizer=Nadam(learning_rate=1e-4),
                  metrics=[coefficient_determination])
predicted = my_model.predict(X_val)
print('Result is saved')
scores = my_model.evaluate(X_val, Y_val, verbose=0)
print('%s: %.2f%%' % (my_model.metrics_names[1], scores[1] * 100))
result = abs(np.mean((predicted - Y_val) ** 2))
print("\nThe mean square error of regression:")
print(result)
predicted = predicted * 3 + 5
Y_origin = Y_val * 3 + 5
MAE = np.mean(abs(predicted - Y_origin))
SD = np.std(abs(predicted - Y_origin))
RMSE = np.sqrt(np.mean((predicted - Y_origin)**2))
R2 = 1-(np.sum((Y_origin - predicted)**2) / np.sum((Y_origin - np.mean(Y_origin))**2))
print('\nprediction MAE: ', MAE)
print('\nprediction SD: ', SD)
print('\nprediction RMSE: ', RMSE)
print('\nprediction R2: ', R2)

