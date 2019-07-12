# Extraction des 4 images

# Image composée de superpixels et chaque super pixel est composé de la façon suivante :
#  0 | 135
# --------
# 45 | 90
# => pour les images de PolaBot

# Image composée de superpixels et chaque super pixel est composé de la façon suivante :
#  45 | 0
# ---------
# 90  | 135
# => pour les images de la caméra polarimétrique

import os
import numpy as np
import imageio
import multiprocessing as mp

# path_folder = "/home/rblin/Documents/Archive/images/PolaBot-Dataset/PolarCam" (PolaBot)
#path_folder = "/home/rblin/Documents/BD_QCAV/train/POLAR"

def process_polar_intensities(path_folder, path_traitement):
    imgs_polar = sorted(os.listdir(path_folder))
    for k in range(len(imgs_polar)):

        image = imageio.imread(path_folder + "/" + imgs_polar[k])

        # On va donc retrouver P0, P45, P90, P135

        P0 = image[0:image.shape[0]:2,1:image.shape[0]:2]
        P45 = image[0:image.shape[0]:2,0:image.shape[0]:2]
        P90 = image[1:image.shape[0]:2,0:image.shape[0]:2]
        P135 = image[1:image.shape[0]:2,1:image.shape[0]:2]

        # Caméra polarimétrique
        imageio.imwrite(path_traitement + str(k) + "_I0.png", P0)
        imageio.imwrite(path_traitement + str(k) + "_I45.png", P45)
        imageio.imwrite(path_traitement + str(k) + "_I90.png", P90)
        imageio.imwrite(path_traitement + str(k) + "_I135.png", P135)

    """"# Calcul des parametres de Stokes

    Stokes = np.zeros((4,P0.shape[0],P0.shape[1]))

    Stokes[0] = P0 + P90
    Stokes[1] = P0 - P90
    Stokes[2] = P45 - P135
    Stokes[3] = 0"""

    """# Polabot
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S0.png", Stokes[0])
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S1.png", Stokes[1])
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S2.png", Stokes[2])"""
    """# Caméra polarimétrique
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S0.png", Stokes[0])
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S1.png", Stokes[1])
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S2.png", Stokes[2])

    # Calcul de l'AOP et du DOP

    AOP = 0.5*np.arctan2(Stokes[1], Stokes[2])

    DOP = np.sqrt(Stokes[2]**2+Stokes[1]**2)/Stokes[0]"""

    """# Polabot
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Params/" + str(k) + "_AOP.png", AOP)
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Params/" + str(k) + "_DOP.png", DOP)"""
    """# Caméra polarimétrique
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_S0.png", Stokes[0])
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_AOP.png", AOP)
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_DOP.png", DOP)"""

path_folder = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h"
path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h_I/"

pool = mp.Pool(mp.cpu_count()-2)
pool.apply(process_polar_intensities(path_folder, path_traitement))
pool.close()