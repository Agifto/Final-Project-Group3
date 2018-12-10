from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

#tensorboard = TensorBoard(log_dir='./my_log_dir', histogram_freq=0, write_graph=True, write_images=False)

# Dimensions of images
img_width = 64
img_height = 64

# Initialize structure parameters
num_train = 18697
num_val = 5621
num_test = 2814
epochs = 5
batch_size = 200

# Assigns input shape depending on whether channels comes first or last
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

# Model definition
model = Sequential()
model.add(Conv2D(16, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (5, 5)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(100, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(800))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Activation('softmax'))

#change loss
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

# Initialize generators with the preprocessing transformations that we will use
train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# # Bringing the data in, resizing it and doing other preprocessing
train_generator = train_datagen.flow_from_directory(directory=r'/home/ubuntu/final_project/Train/', target_size= (img_width, img_height), color_mode = 'rgb', batch_size = batch_size, class_mode = 'categorical', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(directory=r'/home/ubuntu/final_project/Validation/', target_size=(img_width, img_height), color_mode = 'rgb', batch_size=batch_size, class_mode= 'categorical', shuffle=True)

test_generator = test_datagen.flow_from_directory(directory=r'/home/ubuntu/final_project/Test/', target_size= (img_width, img_height), color_mode='rgb', batch_size=batch_size, class_mode= None, shuffle = False)

# Adjust class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_generator.classes),
                                                 train_generator.classes)

class_weight_dict = dict(enumerate(class_weights))

# Train Model
history = model.fit_generator(train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size, class_weight=class_weight_dict)

print('\n')
print(model.summary())

model.save_weights('first_try_weights.h5')
model.save('first_try_model.h5')

# Type this on Terminal in Same Directory as your Keras code
# tensorboard --logdir ./my_log_dir/

#list all data in history
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
