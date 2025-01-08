"""
Train and evaluate the Raman model.
"""
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from datetime import datetime
from model_pH import resnet_34
import scipy.io as scio
from init.prepare import coefficient_determination
from tensorflow.keras.optimizers import Nadam, Adam
np.set_printoptions(threshold=np.inf)

# Initialization
batch_size = 8

checkpoint_save_path = os.path.abspath(os.path.dirname(__file__)) + '/check_point/2d-cnn.ckpt'

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

data_path = os.path.abspath(os.path.dirname(__file__))+'/dataset/data_total.mat'
label_path = os.path.abspath(os.path.dirname(__file__))+'/dataset/pH_total.mat'
data = scio.loadmat(data_path)
label = scio.loadmat(label_path)

X_train=data['train']
Y_train=label['train']
X_valk=data['val']
Y_valk=label['val']
X_val=data['test']
Y_val=label['test']

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
