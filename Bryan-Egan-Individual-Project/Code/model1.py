from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt


'''''
def move_random_files(path_from, path_to, n):
    # function for moving random files from one directory to another (used for creating train and test set)
    files = os.listdir(path_from)
    files.sort()

    for to_move in random.sample(files, int(len(files) * n)):
        f = to_move
        src = path_from + f
        dst = path_to


        #shutil.copy(os.path.join(subdir, to_move), os.path.join(path_to))
        shutil.move(src, dst)

# Separate out 10% of our data from Training set for Testing set
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Chaetognaths/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Chaetognaths/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Crustaceans/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Crustaceans/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Detritus/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Detritus/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Diatoms/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Diatoms/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Gelatinous_Zooplankton/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Gelatinous_Zooplankton/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Other_Invert_Larvae/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Other_Invert_Larvae/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Protists/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Protists/', n=.3)
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Train/Trichodesmium/', path_to='/home/ubuntu/Deep-Learning/final_project/Test/Trichodesmium/', n=.3)

# Separate out 20% of our data from Training set for Validation set
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Chaetognaths/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Chaetognaths/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Crustaceans/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Crustaceans/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Detritus/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Detritus/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Diatoms/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Diatoms/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Gelatinous_Zooplankton/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Gelatinous_Zooplankton/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Other_Invert_Larvae/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Other_Invert_Larvae/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Protists/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Protists/', n=Fraction(2,3))
move_random_files(path_from='/home/ubuntu/Deep-Learning/final_project/Test/Trichodesmium/', path_to='/home/ubuntu/Deep-Learning/final_project/Validation/Trichodesmium/', n=Fraction(2,3))
'''


#test_generator = test_datagen.flow_from_directory(directory=r'/home/ubuntu/Deep-Learning/final_project/Test/Chaetognaths/', target_size= (64, 64), color_mode='grayscale', batch_size=8, class_mode= None, shuffle = False)

#for data_batch, labels_batch in train_generator:
#    print('data batch size:', data_batch.shape)
#    print('labels batch size:', labels_batch.shape)
#    break




img_width = 64
img_height = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=input_shape))
BatchNormalization()
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(32,(3, 3)))
#BatchNormalization(axis=-1)
#model.add(Activation('relu'))
#model.add(Conv2D(32, (3, 3)))
#BatchNormalization(axis=-1)
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
#model.add(Dense(32))
#BatchNormalization()
#model.add(Activation('relu'))

model.add(Dense(8))

model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)
#test_datagen = ImageDataGenerator(rescale = 1 ./255)

train_generator = train_datagen.flow_from_directory(directory=r'C:/Users/Bryan Egan/Desktop/ML2 Cloud/Train/', target_size= (img_width, img_height),
                                                    color_mode = 'rgb',
                                                    batch_size = 32, class_mode = 'categorical', shuffle=True)
validation_generator = validation_datagen.flow_from_directory(directory=r'C:/Users/Bryan Egan/Desktop/ML2 Cloud/Validation/', target_size=(img_width,img_height),
                                                              color_mode = 'rgb', batch_size=32, class_mode= 'categorical', shuffle=True)

#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())


STEP_SIZE_TRAIN=25620//32
STEP_SIZE_VALID=11161//32

training_model = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
)

#keras has a function called history that will show you all the different data files attached to the model
#si from here I am creating a plot that shows accuracy vs validation accuracy and then the losses
print(training_model.history.keys())
#  "Accuracy"
plt.plot(training_model.history['acc'])
plt.plot(training_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(training_model.history['loss'])
plt.plot(training_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()