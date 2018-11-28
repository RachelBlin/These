# Concaténation de P0, P45 et P90 pour avoir des images au même format que les canaux
# RGB afin de pouvoir les passer en entrée à RetinaNet pour détection

import os
import numpy as np
import imageio

path_folder = "/home/rblin/Documents/Traitement_PolaBot/P"

imgs_polar = sorted(os.listdir(path_folder))

for i in range(0, len(imgs_polar), 3):
    im_P_temp = np.zeros((230, 320, 3))
    split_temp = imgs_polar[i].split('_')
    image_P0_temp = imageio.imread(path_folder + "/" + imgs_polar[i])
    image_P45_temp = imageio.imread(path_folder + "/" + imgs_polar[i + 1])
    image_P90_temp = imageio.imread(path_folder + "/" + imgs_polar[i + 2])
    im_P_temp[:, :, 0] = image_P0_temp
    im_P_temp[:, :, 1] = image_P45_temp
    im_P_temp[:, :, 2] = image_P90_temp
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/RetinaNet/" + split_temp[0] + ".png", im_P_temp)
