
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.io as scio
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from datetime import datetime

from init.prepare import coefficient_determination
from tensorflow.keras.optimizers import Adam,Nadam
from models.compared_model import buildmodel

# Initialization
batch_size = 8

checkpoint_save_path = os.path.abspath(os.path.dirname(__file__)) + '/check_point/1d-cnn.ckpt'

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

data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/data_total.mat'
label_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/pH_total.mat'
data = scio.loadmat(data_path)
label = scio.loadmat(label_path)

X_train=data['train']
Y_train=label['train']
X_valk=data['val']
Y_valk=label['val']
X_val=data['test']
Y_val=label['test']


# Train

model=buildmodel(vec)
model.build((8, vec, 1))
model.summary()
model.compile(loss='mean_squared_error',
              optimizer=Nadam(learning_rate=initial_lr),
              metrics=[coefficient_determination])
begin = datetime.now()
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=1,
          validation_data=(X_valk, Y_valk),
          callbacks=[early_stopping, lr_reducer, model_checkpoint])
end = datetime.now()
print('Training time: ', end-begin)


# Evaluation
my_model=buildmodel(vec)
my_model.compile(loss='mean_squared_error',
                 optimizer=Nadam(learning_rate=initial_lr),
                 metrics=[coefficient_determination])
my_model.load_weights(checkpoint_save_path)
predicted = my_model.predict(X_val)
predicted = predicted * 3 + 5
Y_origin = Y_val * 3 + 5

MAE = np.mean(abs(predicted - Y_origin))
SD = np.std(abs(predicted - Y_origin))
RMSE = np.sqrt(np.mean((predicted - Y_origin)**2))
R2 = 1-(np.sum((Y_origin - predicted)**2) / np.sum((Y_origin - np.mean(predicted))**2))
print('\nprediction MAE: ', MAE)
print('\nprediction SD: ', SD)
print('\nprediction RMSE: ', RMSE)
print('\nprediction R2: ', R2)

