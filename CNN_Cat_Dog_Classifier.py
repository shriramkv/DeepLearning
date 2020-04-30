                            ###### Section - 1, Setup ############
from keras.models import Sequential
#  Sequential from keras.models,  This gets our neural network as Sequential network. 
#  As we know, it can be sequential layers or graph

from keras.layers import Conv2D
# We are working with images. All the images are basically 2D. 
# One can go with the 3D if working with videos. 

from keras.layers import MaxPooling2D
# Average Pooling, Sum Pooling and Max Pooling are there. 
# We choose Max pooling. Re collect all what I taught you.

from keras.layers import Flatten
# Well, we must flatten. It is the process of converting all the resultant 2D arrays as single long continuous linear vector.
# This is mandatory, folks.

from keras.layers import Dense
# This is the last step! Yes, full connection of the neural network is performed with this Dense.

from keras.preprocessing.image import ImageDataGenerator
# We are going to use ImageDataGenerator from Keras and hence import it as well! 

                                    # Section 2 - Convolution/Pooling/Flattening/Dense


classifier = Sequential()
# Can we initialize the CNN and start the real coding?

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# We need 2D Convolution. It is 2D image we are dealing with. 
# WE have four arguments to be passed, you know that. 
# First is the number of arguments, We have chosen 32 here. You are free to change the same. 
# Shape of the filter chosen 3 X 3 is mentioned as the second argument. 
# Third argument talks about the input image type, We have RGB (It can be BW also. ) 64 x 64 resolution, 3 refers to RGB
# 4th is the activation function and yes, we stay with ReLU. 

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# As taught, we have chosen Max Pooling with pool size 2 x 2 
classifier.add(Flatten())
# Here, What we are basically doing here is taking the 2-D array,
# i.e pooled image pixels and converting them to a one dimensional single vector.


classifier.add(Dense(units = 128, activation = 'relu'))
# Real fun starts here, we need to create fully connected layer. 
# We have many nodes availbale after flattening and these nodes shall serve as input to the fully connected layers. 
# This layer is present between the input layer and output layer, we can refer to it a hidden layer.
# ‘units’ is number of nodes that should be present in this hidden layer,

classifier.add(Dense(units = 1, activation = 'sigmoid'))
 # Here, it can be relu as well. 
# This is the output and contain only one node, as it is binary classification
# We chose sigmoid function. 

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Compiling the CNN.
# It is time to compile, Yes we specify the following before the same. 
# Optimizer = stochastic gradient descent algorithm.
# loss function to be chosen .
# Metrics remain Accuracy


                             ###### Section - 3, Fitting images with CNN ############
# Can we do some preprocessing, which is done with keras.preprocessing library 
# Remember, it is very structred approach with Keras, all the images in the directory Dogs are considered dogs by KERAS 
# Dont worry too much, we do  flipping, rotating, blurring in the part as preprocessing steps. 

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Deep_DataSet/cat-and-dog/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Deep_DataSet/cat-and-dog/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 80,
epochs = 10,
validation_data = test_set,
validation_steps = 46)

# Here is the most important thing to be learnt! 
# Epochs - What is it? Simple, Epoch is once all the images are proocessed one time individually 
# both forward and backward to the network. 
# Epoch number can be determined by the Trail and Error. 
# More the epoch, better the accuracy, but, it could be overfitting too. 

# Remember -‘steps_per_epoch’ holds the number of training images, i.e the number of images the training_set folder contains.


                                ###### Section - 4, The classification with CNN ############
# We are going to test now! 
# Let us send, cat_or_dog_1.jpg as input to see what the result is. 

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Deep_DataSet/cat-and-dog/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result= classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) 



# We are going to test now! 
# Let us send, cat_or_dog_2.jpg as input to see what the result is. 
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Deep_DataSet/cat-and-dog/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result= classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction) 