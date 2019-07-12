import os
import numpy as np
import imageio
import matplotlib.colors as mc
import colorsys
import cv2

"""path_folder = "/home/rblin/Documents/Aquisitions/Polar/Automne/Ensoleille/FM"
path_traitement_hsv = "/home/rblin/Documents/Aquisitions/HSV/Automne/Ensoleille/FM"
path_traitement_rgb = "/home/rblin/Documents/Aquisitions/RGB_conversion/Automne/Ensoleille/FM"

imgs_polar = sorted(os.listdir(path_folder))"""

path_folder = "/home/rblin/Documents/Aquisitions/Image__2018-12-13__10-37-35.bmp"

#for k in range(len(imgs_polar)):
for k in range(1):

    #image = imageio.imread(path_folder + "/" + imgs_polar[k])
    image = imageio.imread(path_folder)

    # On va donc retrouver P0, P45, P90, P135

    P0 = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)))
    P45 = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)))
    P90 = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)))
    P135 = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)))

    super_pixel = np.zeros((2, 2))
    for i in range(0, image.shape[0], 2):
        for j in range(0, image.shape[1], 2):
            super_pixel = image[i:i + 2, j:j + 2]
            P0[int(i / 2), int(j / 2)] = super_pixel[0, 1]
            P45[int(i / 2), int(j / 2)] = super_pixel[0, 0]
            P90[int(i / 2), int(j / 2)] = super_pixel[1, 0]
            P135[int(i / 2), int(j / 2)] = super_pixel[1, 1]

    # Calcul des paramètres de Stokes

    Stokes = np.zeros((4, P0.shape[0], P0.shape[1]))

    Stokes[0] = P0 + P90
    Stokes[1] = P0 - P90
    Stokes[2] = P45 - P135
    Stokes[3] = 0

    # Calcul de l'AOP et du DOP

    #AOP = 0.5 * np.arctan2(Stokes[1], Stokes[2])

    #DOP = np.sqrt(Stokes[2] ** 2 + Stokes[1] ** 2) / Stokes[0]

    AOP = 0.5*np.arctan(P45-P135, P0 - P90)

    DOP = np.sqrt((P45-P135)**2 + (P0 - P90)**2)/(P0+P45+P90+P135+np.finfo(float).eps*2)

    inten = (P0 + P45 + P90 + P135)/2
    hsv = np.uint8(cv2.merge(((AOP + np.pi/2)/np.pi*180,DOP*255, inten/inten.max()*255)))

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    imageio.imwrite("/home/rblin/Documents/Aquisitions/HSV_autreS.png", hsv)
    imageio.imwrite("/home/rblin/Documents/Aquisitions/RGB_autreS.png", rgb)

    """
    im_temp_hsv = np.zeros((500, 500, 3))
    #im_temp_hsv[:, :, 0] = np.arctan(Stokes[2]/Stokes[1])
    im_temp_hsv[:, :, 0] = (AOP + np.pi)/(2*np.pi) # Valeur comprise entre 0 et 360 soit -180 et 180 mais doit être convertie entre 0 et 1 pour la fonction hsv_to_rgb(), ici AOP est en rad
    im_temp_hsv[:, :, 1] = DOP # Valeur comprise entre 0 et 1 pour hsv_to_rgb()
    im_temp_hsv[:, :, 2] = Stokes[0]/510 # Valeur comprise entre 0 et 1 pour hsv_to_rgb()
    print(im_temp_hsv[250,250,0], im_temp_hsv[250,250,1], im_temp_hsv[250,250,2])

    # Conversion en RGB
    #im_temp_rgb = mc.hsv_to_rgb(im_temp_hsv)
    im_temp_rgb = np.zeros((500,500,3))
    for m in range(500):
        for n in range(500):
            im_temp_rgb[m,n,:] = colorsys.hsv_to_rgb(im_temp_hsv[m,n,0], im_temp_hsv[m,n,1], im_temp_hsv[m,n,2])

    #imageio.imwrite(path_traitement_hsv + "/" + str(k) + "_HSV.png", im_temp_hsv)
    #imageio.imwrite(path_traitement_rgb + "/" + str(k) + "_RGB.png", im_temp_rgb)

    imageio.imwrite("/home/rblin/Documents/Aquisitions/HSV_rachel.png", im_temp_hsv)
    imageio.imwrite("/home/rblin/Documents/Aquisitions/H.png", im_temp_hsv[:,:,0])
    imageio.imwrite("/home/rblin/Documents/Aquisitions/S.png", im_temp_hsv[:,:,1])
    imageio.imwrite("/home/rblin/Documents/Aquisitions/V.png", im_temp_hsv[:,:,2])
    imageio.imwrite("/home/rblin/Documents/Aquisitions/RGB_rachel.png", im_temp_rgb)"""

