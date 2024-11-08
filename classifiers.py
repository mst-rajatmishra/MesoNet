# -*- coding:utf-8 -*-

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = None
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y, batch_size=32, epochs=10, validation_data=None):
        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
    
    def get_accuracy(self, x, y):
        return self.model.evaluate(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso1(Classifier):
    """
    Simple CNN with dropout and batch normalization
    """
    def __init__(self, learning_rate=0.001, dl_rate=1, dropout_rate=0.5):
        self.model = self.init_model(dl_rate, dropout_rate)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
    
    def init_model(self, dl_rate, dropout_rate):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        # First convolutional block
        x1 = Conv2D(16, (3, 3), dilation_rate=dl_rate, strides=1, padding='same', kernel_initializer=HeNormal())(x)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x1 = Conv2D(4, (1, 1), padding='same', activation='relu')(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        # Fully connected layers
        y = Flatten()(x1)
        y = Dropout(dropout_rate)(y)
        y = Dense(1, activation='sigmoid')(y)
        
        return KerasModel(inputs=x, outputs=y)


class Meso4(Classifier):
    """
    Deep CNN with increasing filter sizes and dropout
    """
    def __init__(self, learning_rate=0.001, dropout_rate=0.5):
        self.model = self.init_model(dropout_rate)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def init_model(self, dropout_rate):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))

        # First convolutional block
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer=HeNormal())(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # Second block
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu', kernel_initializer=HeNormal())(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        # Third block
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer=HeNormal())(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        # Fourth block
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer=HeNormal())(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        # Fully connected layers
        y = Flatten()(x4)
        y = Dropout(dropout_rate)(y)
        y = Dense(16)(y)
        y = LeakyReLU(negative_slope=0.1)(y)
        y = Dropout(dropout_rate)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)


class MesoInception4(Classifier):
    """
    Inception-style architecture with multiple convolution filters
    """
    def __init__(self, learning_rate=0.001, dropout_rate=0.5):
        self.model = self.init_model(dropout_rate)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            # First convolution branch
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu', kernel_initializer=HeNormal())(x)
            
            # Second convolution branch
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu', kernel_initializer=HeNormal())(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu', kernel_initializer=HeNormal())(x2)
            
            # Third convolution branch
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu', kernel_initializer=HeNormal())(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)
            
            # Fourth convolution branch
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu', kernel_initializer=HeNormal())(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])
            return y
        return func

    def init_model(self, dropout_rate):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))

        # First Inception block
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # Second Inception block
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        # Third convolutional block
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer=HeNormal())(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        # Fourth convolutional block
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer=HeNormal())(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        # Fully connected layers
        y = Flatten()(x4)
        y = Dropout(dropout_rate)(y)
        y = Dense(16)(y)
        y = LeakyReLU(negative_slope=0.1)(y)
        y = Dropout(dropout_rate)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)
