import os
import imageio
import numpy as np
from shutil import copyfile
#import cv2

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


def get_frames(path_folder):
    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)
    return files

def create_directory(path_directory):
    try:
        os.mkdir(path_directory)
    except FileExistsError:
        print("Directory ", path_directory, " already exists")

def get_final_frames(path_folder, path_directory, first_frame, last_frame, step):
    frames = get_frames(path_folder)
    #create_directory(path_directory)
    new_frames = []
    for i in range(first_frame, last_frame, step):
        #copyfile(path_folder + "/" + frames[i], path_directory + "/" + frames[i])
        new_frames.append(frames[i])
    return new_frames

def process_polar_parameters(path_folder, path_traitement):
    imgs_polar = sorted(os.listdir(path_folder))
    for k in range(110,len(imgs_polar), 50):

        image = imageio.imread(path_folder + "/" + imgs_polar[k])

        I = np.zeros((4, int(image.shape[0]/2), int(image.shape[1]/2)))
        # On va donc retrouver P0, P45, P90, P135

        I[0] = image[0:image.shape[0]:2,1:image.shape[0]:2] / 65520.0 * 255 # I0
        I[1] = image[0:image.shape[0]:2,0:image.shape[0]:2] / 65520.0 * 255 # I45
        I[2] = image[1:image.shape[0]:2,0:image.shape[0]:2] / 65520.0 * 255 # I90
        I[3] = image[1:image.shape[0]:2,1:image.shape[0]:2] / 65520.0 * 255 # I135

        # Caméra polarimétrique
        """if not os.path.exists(path_traitement): # + "I/"):
            os.mkdir(path_traitement) # + "I/")
        imageio.imwrite(path_traitement + imgs_polar[k].split(".")[0] + "_I0.png", I[0]) #"I/" + str(k) + "_I0.png", I[0])
        imageio.imwrite(path_traitement + imgs_polar[k].split(".")[0] + "_I45.png", I[1]) #"I/" + str(k) + "_I45.png", I[1])
        imageio.imwrite(path_traitement + imgs_polar[k].split(".")[0] + "_I90.png", I[2]) #"I/" + str(k) + "_I90.png", I[2])
        imageio.imwrite(path_traitement + imgs_polar[k].split(".")[0] + "_I135.png", I[3]) #"I/" + str(k) + "_I135.png", I[3])"""

        #print(np.amin(I[0]), np.amax(I[0]), np.amin(I[1]), np.amax(I[1]), np.amin(I[2]), np.amax(I[2]), np.amin(I[3]), np.amax(I[3]))

        # Paramètres de Stokes

        Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]))

        Stokes[0] = I[0] + I[2]
        Stokes[1] = I[0] - I[2]
        Stokes[2] = I[1] - I[3]

        Stokes[0] = Stokes[0] / 2
        Stokes[1] = Stokes[1] / 2
        Stokes[2] = Stokes[2] / 2

        #print(I0 + I90)

        #print(np.amin(Stokes[0]), np.amax(Stokes[0]), np.amin(Stokes[1]), np.amax(Stokes[1]), np.amin(Stokes[2]), np.amax(Stokes[2]))
        """if not os.path.exists(path_traitement + "Stokes/"):
            os.mkdir(path_traitement + "Stokes/")
        imageio.imwrite(path_traitement + "Stokes/" + str(k) + "_S0.png", Stokes[0])
        imageio.imwrite(path_traitement + "Stokes/" + str(k) + "_S1.png", Stokes[1])
        imageio.imwrite(path_traitement + "Stokes/" + str(k) + "_S2.png", Stokes[2])"""

        #print("I wrote the stokes parameters\n ---------------------------------------------")

        # Calcul de l'AOP et du DOP

        AOP = (0.5 * np.arctan2(Stokes[2], Stokes[1]) + np.pi/2)/np.pi*255

        DOP = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                if Stokes[0, i,j] <  np.sqrt(np.square(Stokes[2,i,j]) + np.square(Stokes[1,i,j])):
                    DOP[i,j] = 1
                else:
                    DOP[i,j] = np.divide(np.sqrt(np.square(Stokes[2,i,j]) + np.square(Stokes[1,i,j])), Stokes[0,i,j])


        #DOP = np.divide(np.sqrt(np.square(Stokes[2]) + np.square(Stokes[1])), Stokes[0])#*255

        DOP = DOP * 255

        #print(np.amax(0.5 * np.arctan2(Stokes[2], Stokes[1])), np.amin(0.5 * np.arctan2(Stokes[2], Stokes[1])))
        """if not os.path.exists(path_traitement + "Params/"):
            os.mkdir(path_traitement + "Params/")
        imageio.imwrite(path_traitement + "Params/" + str(k) + "_S0.png", Stokes[0])
        imageio.imwrite(path_traitement + "Params/" + str(k) + "_AOP.png", AOP)
        imageio.imwrite(path_traitement + "Params/" + str(k) + "_DOP.png", DOP)"""

        concatenate_frames(I, Stokes, AOP, DOP, path_traitement, k, imgs_polar)


def concatenate_frames(I, Stokes, AOP, DOP, path_traitement, k, imgs_polar):

    # RetinaNet intensities
    im_I04590 = np.zeros((500, 500, 3))
    im_I04590[:, :, 0] = I[0]
    im_I04590[:, :, 1] = I[1]
    im_I04590[:, :, 2] = I[2]
    if not os.path.exists(path_traitement + "I04590/"):
        os.mkdir(path_traitement + "I04590/")
    imageio.imwrite(path_traitement + "I04590/" + imgs_polar[k].split(".")[0] + ".png", im_I04590)

    im_I045135 = np.zeros((500, 500, 3))
    im_I045135[:, :, 0] = I[0]
    im_I045135[:, :, 1] = I[3]
    im_I045135[:, :, 2] = I[1]
    if not os.path.exists(path_traitement + "I013545/"):
        os.mkdir(path_traitement + "I013545/")
    imageio.imwrite(path_traitement + "I013545/" + imgs_polar[k].split(".")[0] + ".png", im_I045135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0]
    im_I090135[:, :, 1] = I[2]
    im_I090135[:, :, 2] = I[3]
    if not os.path.exists(path_traitement + "I090135/"):
        os.mkdir(path_traitement + "I090135/")
    imageio.imwrite(path_traitement + "I090135/" + imgs_polar[k].split(".")[0] + ".png", im_I090135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[1]
    im_I4590135[:, :, 1] = I[2]
    im_I4590135[:, :, 2] = I[3]
    if not os.path.exists(path_traitement + "I4590135/"):
        os.mkdir(path_traitement + "I4590135/")
    imageio.imwrite(path_traitement + "I4590135/" + imgs_polar[k].split(".")[0] + ".png", im_I4590135)

    """im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0] - I[1]
    im_I090135[:, :, 1] = I[0]
    im_I090135[:, :, 2] = I[0] + I[1]
    if not os.path.exists(path_traitement + "RetinaNet_Ieq1/"):
        os.mkdir(path_traitement + "RetinaNet_Ieq1/")
    imageio.imwrite(path_traitement + "RetinaNet_Ieq1/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0] - I[3]
    im_I090135[:, :, 1] = I[0]
    im_I090135[:, :, 2] = I[0] + I[3]
    if not os.path.exists(path_traitement + "RetinaNet_Ieq2/"):
        os.mkdir(path_traitement + "RetinaNet_Ieq2/")
    imageio.imwrite(path_traitement + "RetinaNet_Ieq2/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[1] - I[2]
    im_I090135[:, :, 1] = I[1]
    im_I090135[:, :, 2] = I[1] + I[2]
    if not os.path.exists(path_traitement + "RetinaNet_Ieq3/"):
        os.mkdir(path_traitement + "RetinaNet_Ieq3/")
    imageio.imwrite(path_traitement + "RetinaNet_Ieq3/" + str(k) + ".png", im_I090135)

    im_I090135 = np.zeros((500, 500, 3))
    im_I090135[:, :, 0] = I[0]/I[1]
    im_I090135[:, :, 1] = I[0]/I[2]
    im_I090135[:, :, 2] = I[0]/I[3]
    if not os.path.exists(path_traitement + "RetinaNet_Ieq4/"):
        os.mkdir(path_traitement + "RetinaNet_Ieq4/")
    imageio.imwrite(path_traitement + "RetinaNet_Ieq4/" + str(k) + ".png", im_I090135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]
    im_I4590135[:, :, 1] = I[0]/I[1]
    im_I4590135[:, :, 2] = I[0]/I[2]
    if not os.path.exists(path_traitement + "RetinaNet_eq5/"):
        os.mkdir(path_traitement + "RetinaNet_eq5/")
    imageio.imwrite(path_traitement + "RetinaNet_eq5/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]
    im_I4590135[:, :, 1] = I[0] / I[2]
    im_I4590135[:, :, 2] = I[0] / I[3]
    if not os.path.exists(path_traitement + "RetinaNet_eq6/"):
        os.mkdir(path_traitement + "RetinaNet_eq6/")
    imageio.imwrite(path_traitement + "RetinaNet_eq6/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[1] / I[0]
    im_I4590135[:, :, 1] = I[1] / I[2]
    im_I4590135[:, :, 2] = I[1] / I[3]
    if not os.path.exists(path_traitement + "RetinaNet_eq7/"):
        os.mkdir(path_traitement + "RetinaNet_eq7/")
    imageio.imwrite(path_traitement + "RetinaNet_eq7/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[2] / I[0]
    im_I4590135[:, :, 1] = I[2] / I[1]
    im_I4590135[:, :, 2] = I[2] / I[3]
    if not os.path.exists(path_traitement + "RetinaNet_eq8/"):
        os.mkdir(path_traitement + "RetinaNet_eq8/")
    imageio.imwrite(path_traitement + "RetinaNet_eq8/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[3] / I[0]
    im_I4590135[:, :, 1] = I[3] / I[1]
    im_I4590135[:, :, 2] = I[3] / I[2]
    if not os.path.exists(path_traitement + "RetinaNet_eq9/"):
        os.mkdir(path_traitement + "RetinaNet_eq9/")
    imageio.imwrite(path_traitement + "RetinaNet_eq9/" + str(k) + ".png", im_I4590135)

    im_I4590135 = np.zeros((500, 500, 3))
    im_I4590135[:, :, 0] = I[0]/I[1]
    im_I4590135[:, :, 1] = I[0] / I[2]
    im_I4590135[:, :, 2] = DOP/255
    if not os.path.exists(path_traitement + "RetinaNet_eq10/"):
        os.mkdir(path_traitement + "RetinaNet_eq10/")
    imageio.imwrite(path_traitement + "RetinaNet_eq10/" + str(k) + ".png", im_I4590135)"""

    # retinaNet Stokes
    im_Stokes = np.zeros((500, 500, 3))
    im_Stokes[:, :, 0] = Stokes[0]
    im_Stokes[:, :, 1] = Stokes[1]
    im_Stokes[:, :, 2] = Stokes[2]
    if not os.path.exists(path_traitement + "Stokes/"):
        os.mkdir(path_traitement + "Stokes/")
    imageio.imwrite(path_traitement + "Stokes/" + imgs_polar[k].split(".")[0] + ".png", im_Stokes)

    # RetinaNet Params
    im_Params = np.zeros((500, 500, 3))
    im_Params[:, :, 0] = Stokes[0]
    im_Params[:, :, 1] = AOP
    im_Params[:, :, 2] = DOP
    if not os.path.exists(path_traitement + "Params/"):
        os.mkdir(path_traitement + "Params/")
    imageio.imwrite(path_traitement + "Params/" + imgs_polar[k].split(".")[0] + ".png", im_Params)

    """# HSV image
    HSV = np.zeros((500, 500, 3))
    inten = (I[0] + I[1] + I[2] + I[3]) / 2
    HSV[:, :, 0] = AOP
    HSV[:, :, 1] = DOP
    HSV[:, :, 2] = Stokes[0]
    #print(np.amax(AOP), np.amax(DOP), np.amax(Stokes[0]))
    if not os.path.exists(path_traitement + "HSV/"):
        os.mkdir(path_traitement + "HSV/")
    imageio.imwrite(path_traitement + "HSV/" + imgs_polar[k].split(".")[0] + ".png", HSV)"""

    """# TSV image
    TSV = np.zeros((500, 500, 3))
    TSV[:, :, 0] = AOP
    TSV[:, :, 1] = DOP
    TSV[:, :, 2] = inten / inten.max() * 255
    if not os.path.exists(path_traitement + "RetinaNet_TSV/"):
        os.mkdir(path_traitement + "RetinaNet_TSV/")
    imageio.imwrite(path_traitement + "RetinaNet_TSV/" + str(k) + ".png", TSV)"""

    """# Pauli image
    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0]
    if not os.path.exists(path_traitement + "RetinaNet_Pauli/"):
        os.mkdir(path_traitement + "RetinaNet_Pauli/")
    imageio.imwrite(path_traitement + "RetinaNet_Pauli/" + str(k) + ".png", Pauli)"""

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0] + I[2]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = I[0] - I[2]
    if not os.path.exists(path_traitement + "Pauli2/"):
        os.mkdir(path_traitement + "Pauli2/")
    imageio.imwrite(path_traitement + "Pauli2/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    Pauli = np.zeros((500, 500, 3))
    Pauli[:, :, 0] = I[0]
    Pauli[:, :, 1] = I[1]
    Pauli[:, :, 2] = (I[0]/I[1]) #/ np.amax(I[0] / I[1]) * 255
    if not os.path.exists(path_traitement + "Pauli3/"):
        os.mkdir(path_traitement + "Pauli3/")
    imageio.imwrite(path_traitement + "Pauli3/" + imgs_polar[k].split(".")[0] + ".png", Pauli)

    """Rachel = np.zeros((500, 500, 3))
    Rachel[:, :, 0] = Stokes[0]
    Rachel[:, :, 1] = Stokes[1]
    Rachel[:, :, 2] = DOP
    if not os.path.exists(path_traitement + "RetinaNet_Rachel/"):
        os.mkdir(path_traitement + "RetinaNet_Rachel/")
    imageio.imwrite(path_traitement + "RetinaNet_Rachel/" + str(k) + ".png", Rachel)

    Rachel = np.zeros((500, 500, 3))
    Rachel[:, :, 0] = I[1]
    Rachel[:, :, 1] = I[0]
    Rachel[:, :, 2] = DOP
    if not os.path.exists(path_traitement + "RetinaNet_Rachel2/"):
        os.mkdir(path_traitement + "RetinaNet_Rachel2/")
    imageio.imwrite(path_traitement + "RetinaNet_Rachel2/" + str(k) + ".png", Rachel)"""

def get_rgb_frames(path_folder, path_traitement):
    imgs_rgb = sorted(os.listdir(path_folder))
    for k in range(1223, len(imgs_rgb), 60):
        copyfile(path_folder + imgs_rgb[k], path_traitement + imgs_rgb[k])
#path_folder = "/home/rblin/Documents/BD_QCAV/train/POLAR"
#path_folder = "/home/rblin/Documents/Database_ITSC_correction/test_polar/POLAR"
#path_folder = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/07_05_2019_16h30"
#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/07_05_2019_16h30_I/"
#path_traitement = "/home/rblin/Documents/Database_ITSC_correction/train_polar/"
#path_folder = "/home/rblin/Documents/BD_test/polar"
#path_traitement = "/home/rblin/Documents/BD_test/params_polar/"

#path_folder = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h"
#path_traitement= "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h_I/"

#path_folder = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_16h15"
#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_16h15_I/"

#path_folder = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/20_02_2019_16h30"
#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/20_02_2019_16h30_I/"

#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/DB_100/"

#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/final_07_05_2019_16h30/"

#process_polar_parameters(path_folder, path_traitement)

path_folder = "/home/rblin/Documents/17_05_2019_16h30_rgb/"
path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/final_07_05_2019_16h30/RGB/"

get_rgb_frames(path_folder, path_traitement)