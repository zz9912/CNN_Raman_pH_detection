import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.linear_model import LinearRegression
from init.prepare import get_dataset


vec=287

data_path1 = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/10361_airpls_divide_peak1.mat'
label_path1 = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/10361_ph_label.mat'
data_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/solo_airpls_divide_peak1.mat'
label_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/dataset/solo_ph_label.mat'

X_train, X_val, Y_train, Y_val = get_dataset(data_path, label_path, vec_num=vec,test_size=0.2)
X_train1, X_val1, Y_train1, Y_val1 = get_dataset(data_path1, label_path1, vec_num=vec,test_size=0.2)

X_train = np.concatenate([X_train,X_train1],0)
X_val = np.concatenate([X_val,X_val1],0)
Y_train=np.concatenate([Y_train,Y_train1],0)
Y_val=np.concatenate([Y_val,Y_val1],0)

X_train=np.squeeze(X_train)
X_val=np.squeeze(X_val)

Y_val = Y_val*3+5

model = LinearRegression()
history = model.fit(X_train, Y_train)
predicted = model.predict(X_val)
predicted=predicted*3+5

print(Y_val.shape,predicted.shape)
MAE = np.mean(abs(predicted-Y_val))
SD = np.std(abs(predicted-Y_val))
RMSE = np.sqrt(np.mean((predicted - Y_val)**2))
R2 = 1-(np.sum((Y_val - predicted)**2) / np.sum((Y_val - np.mean(predicted))**2))
print('\nprediction MAE: ', MAE)
print('\nprediction SD: ', SD)
print('\nprediction RMSE: ', RMSE)
print('\nprediction R2_second: ', R2)