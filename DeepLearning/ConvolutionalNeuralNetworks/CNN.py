import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

import time
start_time = time.time()

# Tell tensorflow and keras what to use
import tensorflow as tf
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

classifier = Sequential()

# Help options use Theano backend
# Can add extra convolutional layers and neuron layers
# Filters = number of filter maps
# Kernel size = feature detector size
# input shape = width height and channels of the input image
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation="relu"))

# Halves the size of the feature maps
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add extra convo layer
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation="relu"))

# Halves the size of the feature maps
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps to a single vector
classifier.add(Flatten(data_format="channels_last"))

# Create the actual ANN
classifier.add(Dense(units=128, activation="relu", kernel_initializer="glorot_uniform"))
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="glorot_uniform"))
# softmax for 3 or more classes
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# categorical_crossentropy for 3 more


# Image augmentation by randomly flipping or scaling images in the batch
# Code taken from https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=100,
    class_mode='binary')

tetst_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=100,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=80,
    epochs=25,
    validation_data=tetst_set,
    validation_steps=20)

end_time = time.time()
print("Time taken:", end_time-start_time)
