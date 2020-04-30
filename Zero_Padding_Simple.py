import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
# All the above shall be explained with the code for CNN, Shortly. 

model = Sequential([
    Dense(16, input_shape=(30,30,3), activation='relu'),
    Conv2D(32, kernel_size=(4,4), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(6,6), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
])

# Remember this, Valid as the option for padding, the size will shrink. 
# with same for padding, zero padding happens and dimension gets retained. 
model.summary()




