# Etiquetage des données

import os
import imageio
import pylab as pl
import numpy as np
import random

import keras
from keras.applications.resnet50 import ResNet50
from keras.engine.input_layer import Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

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

# Affichage d'un élément de la ground truth contenant
# une voiture, afin de connaître la valeur des pixels
# qui caractérisent la voiture dans la colormap

path_gt = path + "/" + dossiers[3]
path_gt_car = path_gt + "/Colorized_GroundTruth/1.png"
im = imageio.imread(path_gt_car)
pl.figure(1)
pl.imshow(im)
pl.title("Ground truth contenant une voiture")

# On voit qu'il y a d'autres pixels de la même couleur
# dans l'image ne représentant pas une voiture du coup
# il faudra étiqueter les images à la main

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

# Regardons maintenant le format des images BGR et polar
im_BGR = imageio.imread(path_BGR + "/" + list_img_BGR_train[0])
print("Format image BGR : ", im_BGR.shape)
im_polar = imageio.imread(path_polar + "/" + list_img_polar_train[0])
print("Format image polar : ", im_polar.shape)

# Maintenant qu'on a le format, on va constituer les vecteurs
# qui vont contenir les images pour l'apprentissage du réseau

X_train_BGR = np.zeros((89, 190, 254, 3))
X_train_BGR = X_train_BGR.astype(int)

X_train_polar = np.zeros((89, 460, 640))
X_train_polar = X_train_polar.astype(int)

X_test_BGR = np.zeros((88, 190, 254, 3))
X_test_BGR = X_test_BGR.astype(int)

X_test_polar = np.zeros((88, 460, 640))
X_test_polar = X_test_polar.astype(int)

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
    im_BGR_np_temp = np.asarray(im_BGR_temp)
    im_polar_temp = imageio.imread(path_polar + "/" + list_img_polar_test[n])
    im_polar_np_temp = np.asarray(im_polar_temp)
    X_test_BGR[n, :, :, :] = im_BGR_np_temp
    X_test_polar[n, :, :] = im_polar_np_temp

# Affichage de 9 images au hasard contenant une voiture
# afin de vérifier que l'étiquetage a été fait correctement

itemindex = np.where(Y_train == 0)

indexes = itemindex[0]
indexes = indexes.tolist()

rand_ind = random.sample(indexes, 9)

pl.figure(2)
pl.figure(figsize=(15, 15))
pl.clf()
for i in range(9):
    pl.subplot(3, 3, i + 1)
    im_temp = X_train_BGR[rand_ind[i]]
    pl.imshow(im_temp)

# De même pour les images de test

itemindex = np.where(Y_test==0)

indexes = itemindex[0]
indexes = indexes.tolist()

rand_ind = random.sample(indexes, 9)

pl.figure(3)
pl.figure(figsize=(15,15))
pl.clf()
for i in range(9) :
    pl.subplot(3,3,i+1)
    im_temp = X_test_BGR[rand_ind[i]]
    pl.imshow(im_temp)

# On a maintenant les vecteurs prets pour l'apprentissage,
# mettons donc en place le réseau de neurones

model = ResNet50(weights='imagenet', include_top=False)

#model.summary()

num_classes =2
print(model.input)

input = Input(shape = (190, 254, 3))

x = Flatten()(model(input))
x = Dense(1000, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='linear')(x)

new_model = Model(inputs=input, outputs=x)

new_model.summary()

# Puisque l'on a 89 images de train et de test, il est nécessaire
# d'avoir plus de données pour pouvoir effectuer un apprentissage
# pertinent. Pour cela, on va utiliser des techniques de data
# augmentation.

datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)

# Maintenant que le modèle a été construit pour les
# images RGB, on peut passer à l'entraînement

epochs = 1
batch_size = 16

new_model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

#print(X_test_BGR.shape)
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

#datagen.fit(X_train_BGR)

print(type(X_test_BGR), type(Y_test))

new_model.fit_generator(datagen.flow(X_train_BGR, Y_train,
          batch_size=batch_size),
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_BGR, Y_test))
score = new_model.evaluate(X_test_BGR, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])