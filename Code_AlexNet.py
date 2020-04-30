#!/usr/bin/env python
# coding: utf-8

# In[11]:


import keras

from keras.models import Sequential
#  Sequential from keras.models,  This gets our neural network as Sequential network.
#  As we know, it can be sequential layers or graph

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# Importing, Dense, Activation, Flatten, Activation, Dropout, Conv2D and Maxpooling. 
# Dropout is a technique used to prevent a model from overfitting.

from keras.layers.normalization import BatchNormalization
# For normalization.

import numpy as np

image_shape = (227,227,3)

np.random.seed(1000)
#Instantiate an empty model

model = Sequential()
# It starts here. 

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# First layer has 96 Filters, the input shape is 227 x 227 x 3
# Kernel Size is 11 x 11, Striding 4 x 4, ReLu is the activation function. 

# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer, Here we do flatten! 
model.add(Flatten())

# 1st Fully Connected Layer has 4096 neurons 
model.add(Dense(4096, input_shape=(227*227*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1000))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])


# In[ ]:




