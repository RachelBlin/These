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

# path_folder = "/home/rblin/Documents/Archive/images/PolaBot-Dataset/PolarCam" (PolaBot)
path_folder = "/home/rblin/Documents/BD_QCAV/train/POLAR"

imgs_polar = sorted(os.listdir(path_folder))
# imgs_polar.remove(".DS_Store")

for k in range(3352, len(imgs_polar)):

    image = imageio.imread(path_folder + "/" + imgs_polar[k])

    # On va donc retrouver P0, P45, P90, P135

    P0 = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2)))
    P45 = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2)))
    P90 = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2)))
    P135 = np.zeros((int(image.shape[0]/2), int(image.shape[1]/2)))

    super_pixel = np.zeros((2,2))
    for i in range(0,image.shape[0],2):
        for j in range(0,image.shape[1],2):
            super_pixel = image[i:i+2,j:j+2]
            """
            # Pour PolaBot
            P0[int(i/2), int(j/2)] = super_pixel[0,0]
            P45[int(i/2), int(j/2)] = super_pixel[1,0]
            P90[int(i/2), int(j/2)] = super_pixel[1,1]
            P135[int(i/2), int(j/2)] = super_pixel[0,1]"""
            # Pour les images de la caméra polarimétrique
            P0[int(i / 2), int(j / 2)] = super_pixel[0, 1]
            P45[int(i / 2), int(j / 2)] = super_pixel[0, 0]
            P90[int(i / 2), int(j / 2)] = super_pixel[1, 1]
            P135[int(i / 2), int(j / 2)] = super_pixel[1, 0]

    """# Polabot
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/P/" + str(k) + "_P0.png", P0)
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/P/" + str(k) + "_P45.png", P45)
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/P/" + str(k) + "_P90.png", P90)"""

    path_traitement = "/home/rblin/Documents/BD_QCAV/train/PARAM_POLAR/"

    # Caméra polarimétrique
    imageio.imwrite(path_traitement + "P/" + str(k-1) + "_P0.png", P0)
    imageio.imwrite(path_traitement + "P/" + str(k-1) + "_P45.png", P45)
    imageio.imwrite(path_traitement + "P/" + str(k-1) + "_P90.png", P90)

    # Calcul des parametres de Stokes

    Stokes = np.zeros((4,P0.shape[0],P0.shape[1]))

    Stokes[0] = P0 + P90
    Stokes[1] = P0 - P90
    Stokes[2] = P45 - P135
    Stokes[3] = 0

    """# Polabot
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S0.png", Stokes[0])
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S1.png", Stokes[1])
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Stokes/" + str(k) + "_S2.png", Stokes[2])"""
    # Caméra polarimétrique
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S0.png", Stokes[0])
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S1.png", Stokes[1])
    imageio.imwrite(path_traitement + "Stokes/" + str(k-1) + "_S2.png", Stokes[2])

    # Calcul de l'AOP et du DOP

    AOP = 0.5*np.arctan2(Stokes[1], Stokes[2])

    DOP = np.sqrt(Stokes[2]**2+Stokes[1]**2)/Stokes[0]

    """# Polabot
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Params/" + str(k) + "_AOP.png", AOP)
    imageio.imwrite("/home/rblin/Documents/Traitement_PolaBot/Params/" + str(k) + "_DOP.png", DOP)"""
    # Caméra polarimétrique
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_S0.png", Stokes[0])
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_AOP.png", AOP)
    imageio.imwrite(path_traitement + "Params/" + str(k-1) + "_DOP.png", DOP)