import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

#load pre-shffled train and test data
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3,12,i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))

#image pre-processing
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#splitting of dataset
(x_train,x_valid)= x_train[5000:],x_train[5000:]
(y_train,y_valid)= y_train[5000:],y_train[5000:]

#preparing of labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model setup
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(10,activation = 'softmax'))

# model.summary()

#compiling the model

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics =['accuracy'])

#train the model
checkpointer = ModelCheckpoint(filepath = 'model.weights.best.hdf5', verbose = 1, save_best_only=True)

hist = model.fix(x_train, y_train, batch_size=32, epochs = 100, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose = 2, shuffle = True)
