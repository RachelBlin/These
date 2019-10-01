import os
import imageio
import numpy as np
from shutil import copyfile
#import cv2
import matplotlib.pyplot as plt

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
    dop_max = []
    stokes_zero = []
    rho_one = []
    intensite = []
    equal_i = []
    for k in range(len(imgs_polar)):

        image = imageio.imread(path_folder + "/" + imgs_polar[k])

        I = np.zeros((4, int(image.shape[0]/2), int(image.shape[1]/2)))
        # On va donc retrouver P0, P45, P90, P135
        intens = []
        for i in range(0,image.shape[1],2):
            for j in range(0,image.shape[1],2):
                intens.append((image[i,j+1] + image[i+1,j])/(image[i,j] + image[i+1,j+1]))
                #if abs(image[i,j+1] + image[i+1,j] - (image[i,j] + image[i+1,j+1])) > 26214:
                    #intens += 1
        x = np.asarray(intens)
        bins = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2]
        plt.figure()
        plt.hist(x[np.isfinite(x)], bins)
        plt.title("(I0 + I90)/(I45 + I135)")
        if not os.path.exists(path_traitement + "hist/"):
            os.mkdir(path_traitement + "hist/")
        plt.savefig(path_traitement + "hist/" + imgs_polar[k].split(".")[0] + "_hist.png")

        #intensite.append(intens)

        I[0] = image[0:image.shape[0]:2,1:image.shape[0]:2] / np.max(image[0:image.shape[0]:2,1:image.shape[0]:2]) * 255 # I0
        I[1] = image[0:image.shape[0]:2,0:image.shape[0]:2] / np.max(image[0:image.shape[0]:2,0:image.shape[0]:2]) * 255 # I45
        I[2] = image[1:image.shape[0]:2,0:image.shape[0]:2] / np.max(image[1:image.shape[0]:2,0:image.shape[0]:2]) * 255 # I90
        I[3] = image[1:image.shape[0]:2,1:image.shape[0]:2] / np.max(image[1:image.shape[0]:2,1:image.shape[0]:2]) * 255 # I135

        # Caméra polarimétrique
        if not os.path.exists(path_traitement + "I/"):
            os.mkdir(path_traitement + "I/")
        imageio.imwrite(path_traitement + "I/" + imgs_polar[k].split(".")[0] + "_I0.png", I[0]) #"I/" + str(k) + "_I0.png", I[0])
        imageio.imwrite(path_traitement + "I/" + imgs_polar[k].split(".")[0] + "_I45.png", I[1]) #"I/" + str(k) + "_I45.png", I[1])
        imageio.imwrite(path_traitement + "I/" + imgs_polar[k].split(".")[0] + "_I90.png", I[2]) #"I/" + str(k) + "_I90.png", I[2])
        imageio.imwrite(path_traitement + "I/" + imgs_polar[k].split(".")[0] + "_I135.png", I[3]) #"I/" + str(k) + "_I135.png", I[3])"""

        # Paramètres de Stokes

        Stokes = np.zeros((4, I[0].shape[0], I[0].shape[1]))

        Stokes[0] = I[0] + I[2]
        Stokes[1] = I[0] - I[2]
        Stokes[2] = I[1] - I[3]

        i_as = []
        i_delta_as = []
        for i in range(500):
            for j in range(500):
                I_temp = np.array([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])
                AS = np.array([0.5*(Stokes[0, i, j] + Stokes[1, i, j]), 0.5*(Stokes[0, i, j] + Stokes[2, i, j]), 0.5*(Stokes[0, i, j] - Stokes[1, i, j]), 0.5*(Stokes[0, i, j] - Stokes[2, i, j])])
                i_as.append(np.linalg.norm(I_temp-AS)/(np.linalg.norm(I_temp)+np.linalg.norm(AS)))
                delta_AS = np.array([0.5 * (Stokes[0, i, j] + Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] + 1.125*Stokes[2, i, j]),
                               0.5 * (Stokes[0, i, j] - Stokes[1, i, j]), 0.5 * (Stokes[0, i, j] - 0.875*Stokes[2, i, j])])
                i_delta_as.append(np.linalg.norm(I_temp-delta_AS)/(np.linalg.norm(I_temp)+np.linalg.norm(delta_AS)))

        y = np.asarray(i_as)
        w = np.asarray(i_delta_as)
        #bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
        plt.figure()
        plt.hist(y[np.isfinite(y)], bins)
        plt.title("||I-AS||/(||I|| + ||AS||)")
        if not os.path.exists(path_traitement + "hist/"):
            os.mkdir(path_traitement + "hist/")
        plt.savefig(path_traitement + "hist/" + imgs_polar[k].split(".")[0] + "_hist_ias.png")
        plt.figure()
        plt.hist(w[np.isfinite(w)], bins)
        plt.title("||I-AS||/(||I|| + ||AS||) avec A = A + delta(A)")
        if not os.path.exists(path_traitement + "hist/"):
            os.mkdir(path_traitement + "hist/")
        plt.savefig(path_traitement + "hist/" + imgs_polar[k].split(".")[0] + "_hist_ias_delta.png")

        z = np.count_nonzero(Stokes[0] == 0)

        plt.figure()
        plt.xlim((0, 2))
        plt.plot(x, y, '+')
        plt.title("||I-AS||/(||I|| + ||AS||) en fonction de (I0 + I90)/(I45 + I135)")
        if not os.path.exists(path_traitement + "graphe/"):
            os.mkdir(path_traitement + "graphe/")
        plt.savefig(path_traitement + "graphe/" + imgs_polar[k].split(".")[0] + "rapport.png")

        stokes_zero.append(z)

        """Stokes_aff = np.zeros((4, I[0].shape[0], I[0].shape[1]))

        Stokes_aff[0] = Stokes_aff[0] / (np.max(Stokes_aff[0])) * 255
        Stokes_aff[1] = (Stokes_aff[1] + np.min(Stokes_aff[1])) / (np.max(Stokes_aff[1]) - np.min(Stokes_aff[1])) * 255
        Stokes_aff[2] = (Stokes_aff[2] + np.min(Stokes_aff[2])) / (np.max(Stokes_aff[2]) - np.min(Stokes_aff[2])) * 255

        print(np.max(Stokes))

        #print(I0 + I90)

        #print(np.amin(Stokes[0]), np.amax(Stokes[0]), np.amin(Stokes[1]), np.amax(Stokes[1]), np.amin(Stokes[2]), np.amax(Stokes[2]))
        if not os.path.exists(path_traitement + "Stokes/"):
            os.mkdir(path_traitement + "Stokes/")
        imageio.imwrite(path_traitement + "Stokes/" + imgs_polar[k].split(".")[0] + "_S0.png", Stokes_aff[0])
        imageio.imwrite(path_traitement + "Stokes/" + imgs_polar[k].split(".")[0] + "_S1.png", Stokes_aff[1])
        imageio.imwrite(path_traitement + "Stokes/" + imgs_polar[k].split(".")[0] + "_S2.png", Stokes_aff[2])"""

        #print("I wrote the stokes parameters\n ---------------------------------------------")

        # Calcul de l'AOP et du DOP

        AOP = (0.5 * np.arctan2(Stokes[2], Stokes[1]) + np.pi/2)/np.pi*255
        phi = 0.5 * np.arctan2(Stokes[2], Stokes[1])
        DOP = np.zeros((500, 500))
        rho = np.zeros((500, 500))
        l = 0
        for i in range(500):
            for j in range(500):
                if np.divide(np.sqrt(np.square(Stokes[2,i,j]) + np.square(Stokes[1,i,j])), Stokes[0,i,j]) > 1:
                    #DOP[i,j] = 1
                    #rho[i,j] = 1
                    l += 1
                #else:
                DOP[i,j] = np.divide(np.sqrt(np.square(Stokes[2,i,j]) + np.square(Stokes[1,i,j])), Stokes[0,i,j])
                rho[i,j] = np.divide(np.sqrt(np.square(Stokes[2,i,j]) + np.square(Stokes[1,i,j])), Stokes[0,i,j])
                if DOP[i,j] == 0:
                    equal_i.append([I[0, i, j], I[1, i, j], I[2, i, j], I[3, i, j]])

        rho_one.append(l)

        #DOP = np.divide(np.sqrt(np.square(Stokes[2]) + np.square(Stokes[1])), Stokes[0])#*255

        DOP = DOP / np.max(DOP) * 255
        rho = rho / np.max(rho) * 255

        im_cos = rho * np.cos(phi)
        im_cos = im_cos / np.max(im_cos) * 255
        im_sin = rho * np.sin(phi)
        im_sin = im_sin / np.max(im_sin) * 255

        if not os.path.exists(path_traitement + "CosSin/"):
            os.mkdir(path_traitement + "CosSin/")
        #imageio.imwrite(path_traitement + "Params/" + str(k) + "_S0.png", Stokes[0])
        imageio.imwrite(path_traitement + "CosSin/" + imgs_polar[k].split(".")[0] + "_cos.png", im_cos)
        imageio.imwrite(path_traitement + "CosSin/" + imgs_polar[k].split(".")[0] + "_sin.png", im_sin)

        #print(np.amax(0.5 * np.arctan2(Stokes[2], Stokes[1])), np.amin(0.5 * np.arctan2(Stokes[2], Stokes[1])))
        if not os.path.exists(path_traitement + "Params/"):
            os.mkdir(path_traitement + "Params/")
        #imageio.imwrite(path_traitement + "Params/" + str(k) + "_S0.png", Stokes[0])
        imageio.imwrite(path_traitement + "Params/" + imgs_polar[k].split(".")[0] + "_AOP.png", AOP)
        imageio.imwrite(path_traitement + "Params/" + imgs_polar[k].split(".")[0] + "_DOP.png", DOP)

        Max = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                #print(DOP[i,j], AOP[i,j], Stokes[1,i,j], Stokes[2,i,j], I[0,i,j])
                Max[i,j] = max(DOP[i,j], AOP[i,j], Stokes[1,i,j], Stokes[2,i,j], I[0,i,j])

        if not os.path.exists(path_traitement + "Max_fusion/"):
            os.mkdir(path_traitement + "Max_fusion/")
        imageio.imwrite(path_traitement + "Max_fusion/" + imgs_polar[k].split(".")[0] + "_max.png", Max)

        Min = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                # print(DOP[i,j], AOP[i,j], Stokes[1,i,j], Stokes[2,i,j], I[0,i,j])
                Min[i, j] = min(DOP[i, j], AOP[i, j], Stokes[1, i, j], Stokes[2, i, j], I[0, i, j])

        if not os.path.exists(path_traitement + "Min_fusion/"):
            os.mkdir(path_traitement + "Min_fusion/")
        imageio.imwrite(path_traitement + "Min_fusion/" + imgs_polar[k].split(".")[0] + "_min.png", Min)
        dop_max.append(np.max(DOP))
    print("DOP max pour chaque image : ", dop_max)
    print("Images polarimétriques : ", imgs_polar)
    #print("Nombre de pixels pour lesquels I0 + I90 != I45 + I135 : ", intensite)
    print("Nombre de pixels pour lesquels S0 = 0 : ", stokes_zero)
    print("Nombre de pixels pour lesquels DOP > 1 : ", rho_one)
    print("Valeurs des intensités pour DOP=0 : ", equal_i)

        #concatenate_frames(I, Stokes, AOP, DOP, path_traitement, k, imgs_polar)



def concatenate_frames(I, Stokes, AOP, DOP, path_traitement, k, imgs_polar):

    """# RetinaNet intensities
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
    imageio.imwrite(path_traitement + "I4590135/" + imgs_polar[k].split(".")[0] + ".png", im_I4590135)"""

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
    for k in range(0, len(imgs_rgb), 50*4):
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

path_folder = "/media/rblin/EC42-B858/test_polar/Raw/"
#path_traitement = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/final_07_05_2019_16h30/RGB/"
path_traitement = "/media/rblin/EC42-B858/test_polar/Process/"

#get_rgb_frames(path_folder, path_traitement)
process_polar_parameters(path_folder, path_traitement)