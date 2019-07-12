import os
from shutil import copyfile

def sort_label_imgs(path_images, path_labels, path_final_images, path_final_labels):
    try:
        labels = sorted(os.listdir(path_labels))
    except FileNotFoudError:
        print("No such file or directory ", path_labels)

    try:
        images = sorted(os.listdir(path_images + "RetinaNet_I04590/"))
    except FileNotFoudError:
        print("No such file or directory ", path_images)

    if not os.path.exists(path_final_images + "I04590/"):
        os.mkdir(path_final_images + "I04590/")

    if not os.path.exists(path_final_images + "I045135/"):
        os.mkdir(path_final_images + "I045135/")

    if not os.path.exists(path_final_images + "I090135/"):
        os.mkdir(path_final_images + "I090135/")

    if not os.path.exists(path_final_images + "I4590135/"):
        os.mkdir(path_final_images + "I4590135/")

    if not os.path.exists(path_final_images + "Params/"):
        os.mkdir(path_final_images + "Params/")

    if not os.path.exists(path_final_images + "Pauli2/"):
        os.mkdir(path_final_images + "Pauli2/")

    if not os.path.exists(path_final_images + "Pauli3/"):
        os.mkdir(path_final_images + "Pauli3/")

    if not os.path.exists(path_final_images + "Stokes/"):
        os.mkdir(path_final_images + "Stokes/")

    if not os.path.exists(path_final_images + "Rachel/"):
        os.mkdir(path_final_images + "Rachel/")

    if not os.path.exists(path_final_images + "Rachel2/"):
        os.mkdir(path_final_images + "Rachel2/")


    k = 3351
    while k <len(images):
        if str(k) + ".xml" in labels:
            copyfile(path_images + "RetinaNet_I04590/" + str(k) + ".png",
                     path_final_images + "I04590/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I045135/" + str(k) + ".png",
                     path_final_images + "I045135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I090135/" + str(k) + ".png",
                     path_final_images + "I090135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_I4590135/" + str(k) + ".png",
                     path_final_images + "I4590135/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Params/" + str(k) + ".png",
                     path_final_images + "Params/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Pauli2/" + str(k) + ".png",
                     path_final_images + "Pauli2/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Pauli3/" + str(k) + ".png",
                     path_final_images + "Pauli3/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Stokes/" + str(k) + ".png",
                     path_final_images + "Stokes/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Rachel/" + str(k) + ".png",
                     path_final_images + "Rachel/" + str(k) + ".png")
            copyfile(path_images + "RetinaNet_Rachel2/" + str(k) + ".png",
                     path_final_images + "Rachel2/" + str(k) + ".png")
            copyfile(path_labels + str(k) + ".xml",
                     path_final_labels + str(k) + ".xml")
        k = k+1
        print(k)

def remove_bad_images(path_images):
    images = sorted(os.listdir(path_images))
    for k in range(len(images)):
        #if images[k][0] != '0':
        os.remove(path_images + images[k])



path_images = "/home/rblin/Documents/Database_ITSC_correction/train_polar/"
#path_labels = "/home/rblin/Documents/POLARIMETRIC_DB_3239/test_polar/LABELS/"
path_labels = "/home/rblin/Documents/BD_QCAV/train/LABELS/PARAM_POLAR/"
path_final_images = "/home/rblin/Documents/New_polarimetric_DB_3239/train_polar/PARAM_POLAR/"
path_final_labels = "/home/rblin/Documents/New_polarimetric_DB_3239/train_polar/LABELS/"

#path_images = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/07_05_2019_16h30_I/"
#path_images = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h_I/"
#path_images = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_16h15_I/"
#path_images = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/20_02_2019_16h30_I/"

sort_label_imgs(path_images, path_labels, path_final_images, path_final_labels)

#remove_bad_images(path_images)