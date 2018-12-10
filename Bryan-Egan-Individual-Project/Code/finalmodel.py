# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 07:43:02 2018

@author: Bryan Egan
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

img_width = 64
img_height = 64

# Initialize structure parameters
num_train = 25402
num_val = 6640
num_test = 2814
epochs = 5
batch_size = 200

# Assigns input shape depending on whether channels comes first or last
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Model definition
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('sigmoid'))

#change loss
model.compile(loss='mean_squared_error',
              optimizer='nadam',
              metrics=['accuracy'])

# Initialize generators with the preprocessing transformations that we will use 
train_datagen = ImageDataGenerator(rescale = 1./255, featurewise_center=False, samplewise_center=False, featurewise_std_normalization=True, samplewise_std_normalization=False, zca_whitening=True, zca_epsilon=1e-06)
#train_datagen = ImageDataGenerator(rescale = 1./255, featurewise_center=True, featurewise_std_normalization=True)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# # Bringing the data in, resizing it and doing other preprocessing
train_generator = train_datagen.flow_from_directory(directory=r'C:/Users/Bryan Egan/Desktop/ML2 Cloud/Train/', target_size= (img_width, img_height), color_mode = 'rgb', batch_size = batch_size, class_mode = 'categorical', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(directory=r'C:/Users/Bryan Egan/Desktop/ML2 Cloud/Validation/', target_size=(img_width, img_height), color_mode = 'rgb', batch_size=batch_size, class_mode= 'categorical', shuffle=True)

test_generator = test_datagen.flow_from_directory(directory=r'C:/Users/Bryan Egan/Desktop/ML2 Cloud/Test/', target_size= (img_width, img_height), color_mode='rgb', batch_size=batch_size, class_mode= None, shuffle = False)

# Adjust class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_generator.classes),
                                                 train_generator.classes)

class_weight_dict = dict(enumerate(class_weights))

# Fit model
history = model.fit_generator(train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size, class_weight=class_weight_dict)
 
print(class_weight_dict)
print('\n')
print(model.summary())

# Type this on Terminal in Same Directory as your Keras code
# tensorboard --logdir ./my_log_dir/

model.save_weights('first_try.h5')

# List all data in history
print(history.history.keys())

# Plot accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper right')
plt.show()