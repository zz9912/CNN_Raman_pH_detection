from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Add,  Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Add
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def basic_block(input,f):
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input])
    x = ReLU()(x)
    return x

def basic_down(input,f):
    x_add = Conv2D(filters=f, kernel_size=1,strides=2,kernel_initializer='he_normal')(input)
    x = Conv2D(filters=f, kernel_size=3,strides=2, padding="SAME",kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=f, kernel_size=3, padding="SAME",kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_add])
    x = ReLU()(x)
    return x

def resnet_34(vec,dim=[3,4,6,3]):
    input_spectra = Input(shape=(vec, 1))
    input_spectra1=tf.abs(input_spectra-tf.transpose(input_spectra,perm=[0,2,1]))
    input_spectra1 = tf.expand_dims(input_spectra1, axis=-1)
    f1=64
    f2=128
    f3=256
    f4=512
    x = Conv2D(filters=64,kernel_size=7,strides=2,padding='same',kernel_initializer='he_normal')(input_spectra1)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    for _ in range(dim[0]):
        x = basic_block(x,f1)

    x = basic_down(x,f2)
    for _ in range(dim[1]-1):
        x = basic_block(x,f2)

    x = basic_down(x, f3)
    for _ in range(dim[2] - 1):
        x = basic_block(x, f3)

    x = basic_down(x, f4)
    for _ in range(dim[3] - 1):
        x = basic_block(x, f4)

    x = GlobalAveragePooling2D()(x)

    _output = Dense(1, kernel_initializer=initializers.RandomUniform)(x)
    model = Model(inputs=input_spectra, outputs=_output)

    return model

