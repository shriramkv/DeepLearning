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
traindata = trdata.flow_from_directory(directory='C:/Deep_DataSet/cat-and-dog/training_set',target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory='C:/Deep_DataSet/cat-and-dog/test_set', target_size=(224,224))
# Can we assign the test and training images. As usual, we can take 70/30

                                        # Section 2 - Convolution/Pooling/Flattening/Dense

VGG = keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')
VGG.trainable = False
#model = Sequential()
# Can we initialize the CNN and start the real coding
# Here, What we are basically doing here is taking the 2-D array,
# i.e pooled image pixels and converting them to a one dimensional single vector.

# We have 1 x Dense layer of 4096 units
# We have 1 x Dense layer of 4096 units
# We have 1 x Dense Softmax layer of 2 units
model = keras.Sequential([
  VGG,
  keras.layers.Flatten(),
  keras.layers.Dense(units=256,activation="relu"),
  keras.layers.Dense(units=256,activation="relu"),
  keras.layers.Dense(units=2, activation="softmax")
])
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# Can we compile??
# Specify all the arugyments, metrics clearly.

model.summary()
# We get a summary out here. It is a table, folks!

hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model.save('vggclf.h5')

# Can we visualize?
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

# We can test with first input image - It is dog!
from keras.preprocessing import image
img = image.load_img("C:/Deep_DataSet/cat-and-dog/cat_or_dog_1.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("vggclf.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')