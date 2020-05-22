from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()

model.add(Flatten(input_shape = (28,28)))

model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()