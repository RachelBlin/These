# Concaténation de P0, P45 et P90 pour avoir des images au même format que les canaux
# RGB afin de pouvoir les passer en entrée à RetinaNet pour détection

import os
import numpy as np
import imageio

# path_folder = "/home/rblin/Documents/Traitement_PolaBot/P" (Polabot)

path_traitement = "/home/rblin/Documents/Aquisitions/Traitement_polar/Hiver/Brouillard/DM/"
path_folder_P = path_traitement + "P"

imgs_polar_P = sorted(os.listdir(path_folder_P))

for i in range(0, len(imgs_polar_P), 3):
    # im_P_temp = np.zeros((230, 320, 3)) (Polabot)
    im_P_temp = np.zeros((500, 500, 3))
    split_temp = imgs_polar_P[i].split('_')
    image_P0_temp = imageio.imread(path_folder_P + "/" + imgs_polar_P[i])
    image_P45_temp = imageio.imread(path_folder_P + "/" + imgs_polar_P[i + 1])
    image_P90_temp = imageio.imread(path_folder_P + "/" + imgs_polar_P[i + 2])
    im_P_temp[:, :, 0] = image_P0_temp
    im_P_temp[:, :, 1] = image_P45_temp
    im_P_temp[:, :, 2] = image_P90_temp
    # imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/RetinaNet/" + split_temp[0] + ".png", im_P_temp) (Polabot)
    imageio.imwrite(path_traitement + "RetinaNet_P/" + split_temp[0] + ".png", im_P_temp)

path_folder_Stokes = path_traitement + "Stokes"

imgs_polar_Stokes = sorted(os.listdir(path_folder_Stokes))

for i in range(0, len(imgs_polar_Stokes), 3):
    im_Stokes_temp = np.zeros((500, 500, 3))
    split_temp = imgs_polar_Stokes[i].split('_')
    image_S0_temp = imageio.imread(path_folder_Stokes + "/" + imgs_polar_Stokes[i])
    image_S1_temp = imageio.imread(path_folder_Stokes + "/" + imgs_polar_Stokes[i + 1])
    image_S2_temp = imageio.imread(path_folder_Stokes + "/" + imgs_polar_Stokes[i + 2])
    im_Stokes_temp[:, :, 0] = image_S0_temp
    im_Stokes_temp[:, :, 1] = image_S1_temp
    im_Stokes_temp[:, :, 2] = image_S2_temp
    imageio.imwrite(path_traitement + "RetinaNet_Stokes/" + split_temp[0] + ".png", im_Stokes_temp)

path_folder_Params = path_traitement + "Params"

imgs_polar_Params = sorted(os.listdir(path_folder_Params))

for i in range(0, len(imgs_polar_Params), 3):
    im_Params_temp = np.zeros((500, 500, 3))
    split_temp = imgs_polar_Params[i].split('_')
    image_1_temp = imageio.imread(path_folder_Params + "/" + imgs_polar_Params[i])
    image_2_temp = imageio.imread(path_folder_Params + "/" + imgs_polar_Params[i + 1])
    image_3_temp = imageio.imread(path_folder_Params + "/" + imgs_polar_Params[i + 2])
    im_Params_temp[:, :, 0] = image_3_temp
    im_Params_temp[:, :, 1] = image_1_temp
    im_Params_temp[:, :, 2] = image_2_temp
    imageio.imwrite(path_traitement + "RetinaNet_Params/" + split_temp[0] + ".png", im_Params_temp)