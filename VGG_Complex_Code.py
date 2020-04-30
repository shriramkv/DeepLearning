#!/usr/bin/env python
# coding: utf-8

# In[11]:


import keras,os
from keras.models import Sequential
#  Sequential from keras.models,  This gets our neural network as Sequential network.
#  As we know, it can be sequential layers or graph

from keras.layers import Conv2D
# We are working with images. All the images are basically 2D.
# One can go with the 3D if working with videos.

from keras.layers import MaxPool2D
# Average Pooling, Sum Pooling and Max Pooling are there.
# We choose Max pooling. Re collect all what I taught you. from keras.layers import  


from keras.layers import Flatten
# Well, we must flatten. It is the process of converting all the resultant 2D arrays as single long continuous linear vector.
# This is mandatory, folks.

from keras.layers import Dense
# This is the last step! Yes, full connection of the neural network is performed with this Dense.

from keras.preprocessing.image import ImageDataGenerator
# We are going to use ImageDataGenerator from Keras and hence import it as well! It helps in rescale, rotate, zoom, flip etc.

import numpy as np
# Yes, Numpy matters too!

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="C:/Deep_DataSet/cat-and-dog/training_set",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="C:/Deep_DataSet/cat-and-dog/test_set", target_size=(224,224))
# Can we assign the test and training images. As usual, we can take 70/30

                                        # Section 2 - Convolution/Pooling/Flattening/Dense
model = Sequential()
# Can we initialize the CNN and start the real coding?


model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# We need 2D Convolution. It is 2D image we are dealing with. (All these are specs you need to follow - No other go!)
# We have four arguments to be passed, you know that.
# Specify input shape (224, 224, 3), Number of filters, Filter size and activation function.  
# Remember this, Filter == kernel. Number of filters is specified along with dimension.

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Follow the same procedure

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Follow the same procedure


model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Follow the same procedure



model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Follow the same procedure
model.load_weights('vggweights.h5')
for layer in model.layers:
    layer.trainable=False
model.add(Flatten())
# Here, What we are basically doing here is taking the 2-D array,
# i.e pooled image pixels and converting them to a one dimensional single vector.

model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=2, activation="softmax"))
# We have 1 x Dense layer of 4096 units
# We have 1 x Dense layer of 4096 units
# We have 1 x Dense Softmax layer of 2 units


model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# Can we compile??
# Specify all the arugyments, metrics clearly.

model.summary()
# We get a summary out here. It is a table, folks!

#from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')

# The model is ready now. We should import ModelCheckpoint and EarlyStopping method from keras.
# ModelCheckpoint helps us to save the model by monitoring a specific parameter of the model. Here, we monitor accuracy val_acc.
# Only if there is an improvement in the accuracy, the model gets stored.
# EarlyStopping helps us to stop the training of the model early if there is no increase in the parameter
# We have set patience as  20 which means that the model will stop to train if it doesn’t see any rise in validation accuracy in 5 epochs.

#hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5,callbacks=[checkpoint,early])
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model.save('weights1')

# Can we visualize?
import matplotlib.pyplot as plt
#plt.plot(hist.history["acc"])
#plt.plot(hist.history['val_acc'])
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title("model accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
#plt.show()

# We can test with first input image - It is dog!
from keras.preprocessing import image
img = image.load_img("C:/Deep_DataSet/cat-and-dog/cat_or_dog_1.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("weights1")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')


# In[ ]:





# In[ ]:




