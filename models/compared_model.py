from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, PReLU
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Model

def buildmodel(vec):

    input_layer = layers.Input(shape=(vec, 1))

    # Block 1
    x = BatchNormalization()(input_layer)
    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Block 2
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Block 3
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Output layer
    output_layer = Dense(1, activation='sigmoid', kernel_initializer='normal')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
