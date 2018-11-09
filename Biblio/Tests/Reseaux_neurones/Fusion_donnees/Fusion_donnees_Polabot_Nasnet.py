import os
import imageio
import numpy as np
import random
import cv2

import keras
from keras.applications.nasnet import NASNetLarge
from keras.engine.input_layer import Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, SGD

# Définition des chemins
path = "/Users/rblin/Downloads/PolaBot-Dataset"
dossiers = os.listdir(path)
path_BGR = path + "/" + dossiers[1]
path_polar = path + "/" + dossiers[5]

# Liste de éléments
dossiers = os.listdir(path)
elements_BGR = sorted(os.listdir(path_BGR))
elements_polar = sorted(os.listdir(path_polar))

# Retirer les éléments qui ne sont pas des images
elements_BGR.remove("output")
elements_BGR.remove(".DS_Store")
elements_polar.remove(".DS_Store")

Y = np.zeros((177, 1))
Y = Y.astype(int)

# Indices des images ne contenant pas de voiture

ind = [19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 39, 40, 41, 42, 43, 44, 46, 47, 51, 54, 55, 56, 59, 62, 66,
       67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 98,
       99, 100, 101, 102, 103, 104, 105, 106, 109, 114, 115, 119, 120, 129, 133, 148, 149, 153, 164, 166, 167, 169, 170,
       171, 172, 173, 174, 175, 176]

Y[ind] = 1

# Il faut maintenant diviser les images en ensembles
# d'apprentissage et de test

indexes_train = random.sample(range(0, 176), 89)
indexes_test = [i for i in range(177)]
for j in indexes_train:
    indexes_test.remove(j)

Y_train = Y[indexes_train]
Y_test = Y[indexes_test]

list_img_BGR_train = []
list_img_polar_train = []
for k in indexes_train:
    list_img_BGR_train.append(elements_BGR[k])
    list_img_polar_train.append(elements_polar[k])

list_img_BGR_test = []
list_img_polar_test = []
for l in indexes_test:
    list_img_BGR_test.append(elements_BGR[l])
    list_img_polar_test.append(elements_polar[l])

# Constitution des vecteurs qui vont contenir
# les images pour l'apprentissage du réseau

X_train_BGR = np.zeros((89, 190, 254, 3))
X_train_BGR = X_train_BGR.astype(int)

X_train_polar = np.zeros((89, 460, 640))
X_train_polar = X_train_polar.astype(int)

X_test_BGR = np.zeros((88, 190, 254, 3))
X_test_BGR = X_test_BGR.astype(int)

X_test_polar = np.zeros((88, 460, 640))
X_test_polar = X_test_polar.astype(int)

# Images redimensionnées en 224*224*3 pour être testées par NasNet

X_test_BGR_resized = np.zeros((88, 331, 331, 3))
X_test_BGR_resized = X_test_BGR_resized.astype(int)

# Images de train
for m in range(len(list_img_BGR_train)):
    im_BGR_temp = imageio.imread(path_BGR + "/" + list_img_BGR_train[m])
    im_BGR_np_temp = np.asarray(im_BGR_temp)
    im_polar_temp = imageio.imread(path_polar + "/" + list_img_polar_train[m])
    im_polar_np_temp = np.asarray(im_polar_temp)
    X_train_BGR[m, :, :, :] = im_BGR_np_temp
    X_train_polar[m, :, :] = im_polar_np_temp

for n in range(len(list_img_BGR_test)):
    im_BGR_temp = imageio.imread(path_BGR + "/" + list_img_BGR_test[n])
    im_BGR_resized_temp = cv2.resize(im_BGR_temp, (331,331))
    im_BGR_resized_np_temp = np.asarray(im_BGR_resized_temp)
    im_BGR_np_temp = np.asarray(im_BGR_temp)
    im_polar_temp = imageio.imread(path_polar + "/" + list_img_polar_test[n])
    im_polar_np_temp = np.asarray(im_polar_temp)
    X_test_BGR[n, :, :, :] = im_BGR_np_temp
    X_test_polar[n, :, :] = im_polar_np_temp
    X_test_BGR_resized[n, :, :, :] = im_BGR_resized_np_temp

X_train_BGR = X_train_BGR.astype('float32')
X_train_polar = X_train_polar.astype('float32')
X_test_BGR = X_test_BGR.astype('float32')
X_test_polar = X_test_polar.astype('float32')
X_test_BGR_resized = X_test_BGR_resized.astype('float32')
X_train_BGR /= 255
X_train_polar /= 255
X_test_BGR /= 255
X_test_polar /= 255
X_test_BGR_resized /=225

X_train_BGR = (X_train_BGR - 0.5) * 2
X_train_polar = (X_train_polar - 0.5) * 2
X_test_BGR = (X_test_BGR - 0.5) * 2
X_test_polar = (X_test_polar - 0.5) * 2
X_test_BGR_resized = (X_test_BGR_resized - 0.5) * 2

# On a maintenant les vecteurs prets pour l'apprentissage,
# mettons donc en place le réseau de neurones

model = NASNetLarge(weights='imagenet', include_top=True, classes=1000)

"""num_classes =2
print(model.input)

#input = Input(shape = (190, 254, 3))

x = Flatten()(model(input))
x = Dense(1000, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)

new_model = Model(inputs=input, outputs=x)

new_model.summary()

# Puisque l'on a 89 images de train et de test, il est nécessaire
# d'avoir plus de données pour pouvoir effectuer un apprentissage
# pertinent. Pour cela, on va utiliser des techniques de data
# augmentation.

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='nearest')

# Maintenant que le modèle a été construit pour les
# images RGB, on peut passer à l'entraînement

epochs = 5
batch_size = 16

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

new_model.compile(loss=categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

train_generator = datagen.flow(
        X_train_BGR,
        Y_train,
        batch_size=batch_size)

validation_generator = datagen.flow(
        X_test_BGR,
        Y_test,
        batch_size=batch_size)

new_model.fit_generator(train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)"""
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss=categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])
#score = model.evaluate(X_test_BGR, Y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

predictions = model.predict(X_test_BGR_resized[0:10,:,:,:])
index_predict = np.argmax(predictions, axis=1)
unique, counts = np.unique(index_predict, return_counts=True)
print(dict(zip(unique, counts)))

# Pour la classification de 10 images, on retrouve les classes suivantes :
