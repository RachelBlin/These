import os
import numpy as np
import imageio
import matplotlib.colors as mc
import colorsys

path_folder = "/home/rblin/Documents/images_test/reflets/4.jpg"

image_rgb = imageio.imread(path_folder)

image_hsv = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3))
for m in range(image_rgb.shape[0]):
    for n in range(image_rgb.shape[1]):
        image_hsv[m, n, :] = colorsys.rgb_to_hsv(image_rgb[m,n,0], image_rgb[m,n,1], image_rgb[m,n,2])

imageio.imwrite("/home/rblin/Documents/images_test/HSV.png", image_hsv)

image_rgb_back = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3))
for m in range(image_rgb.shape[0]):
    for n in range(image_rgb.shape[1]):
        image_rgb_back[m, n, :] = colorsys.hsv_to_rgb(image_hsv[m,n,0], image_hsv[m,n,1], image_hsv[m,n,2])

imageio.imwrite("/home/rblin/Documents/images_test/RGB.png", image_rgb_back)