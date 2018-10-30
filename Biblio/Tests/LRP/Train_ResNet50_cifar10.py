from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.insert(0, os.path.abspath('..'))

import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam

import numpy as np

import cv2

NUM_CLASSES = 10
BATCH_SIZE = 2
NUM_EPOCHS = 3
use_data_aug = True

# img_arr is of shape (n, h, w, c)
def resize_image_arr(img_arr):
    x_resized_list = []
    for i in range(img_arr.shape[0]):
        img = img_arr[0]
        resized_img = cv2.resize(img, (224, 224))
        x_resized_list.append(resized_img)
    return np.stack(x_resized_list)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
y_train = y_train
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
y_test = y_test

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model3 = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')

# add a global spatial average pooling layer
x = model3.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- 10 classes for CIFAR10
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model4 = Model(inputs=model3.input, outputs=predictions)


opt = keras.optimizers.rmsprop(lr=0.1, decay=0.5)

model4.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model4.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test), shuffle=False)

model4.save('resnet50_cifar10.h5')