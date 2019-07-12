import os
import shutil

def rename_frames_in_order(path_folder):
    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)

    for f in files:#[:2]:
        #print(f)
        name = f.split("_")
        #print(name[3])
        if len(name[3])==9:
            name[3] = '000' + name[3]
        elif len(name[3])==10:
            name[3] = '00' + name[3]
        elif len(name[3])==11:
            name[3] = '0' + name[3]
        #print(name[3])
        f_new = name[3]
        #print(path_folder + f)
        #print(path_folder + f_new)
        os.rename(path_folder + f, path_folder + f_new)

def rename_frames_in_order_param(path_folder):
    try:
        files = sorted(os.listdir(path_folder))
    except FileNotFoudError:
        print("No such file or directory ", path_folder)

    for f in files:
        name = f.split("_")
        if len(name[0])==1:
            name[0] = '000000' + name[0]
        elif len(name[0])==2:
            name[0] = '00000' + name[0]
        elif len(name[0])==3:
            name[0] = '0000' + name[0]
        elif len(name[0]) == 4:
            name[0] = '000' + name[0]
        elif len(name[0]) == 5:
            name[0] = '00' + name[0]
        elif len(name[0])==6:
            name[0] = '0' + name[0]
        #print(name[3])
        f_new = name[0]
        #print(f_new)
        #print(path_folder + f_new + "_" + name[1])
        #print(path_folder + f)
        #print(path_folder + f_new)
        os.rename(path_folder + f, path_folder + f_new + "_" + name[1])

"""def rename_sequence(path_folders):
    files = sorted(os.listdir(path_folders[0]))
    len = len(files) + 1
    for path in path_folders[1:]:
        files = sorted(os.listdir(path))
        for f in files:
            name = f.split('.')
            int = """

def rename_rgb(path_rgb_movie):
    sequences = sorted(os.listdir(path_rgb_movie))
    len_seq = 0
    for seq in sequences:
        files = os.listdir(path_rgb_movie + seq)
        if len(files) >=1:
            for f in files:
                name = f.split("frame")
                frame_number = name[1].split(".")
                nb = str(int(frame_number[0]) + 41857)
                if len(nb) == 1:
                    nb = '000000' + nb
                elif len(nb) == 2:
                    nb = '00000' + nb
                elif len(nb) == 3:
                    nb = '0000' + nb
                elif len(nb) == 4:
                    nb = '000' + nb
                elif len(nb) == 5:
                    nb = '00' + nb
                elif len(nb) == 6:
                    nb = '0' + nb
                os.rename(path_rgb_movie + seq + "/" + f, path_rgb_movie + seq + "/" + nb + ".png")
            len_seq = len(files)

def move_rgb(path_rgb_movie, final_path):
    sequences = sorted(os.listdir(path_rgb_movie))
    for seq in sequences:
        files = os.listdir(path_rgb_movie + seq)
        for f in files:
            shutil.move(path_rgb_movie + seq + "/" + f, final_path + f)


#path_1 = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/07_05_2019_16h30/"
#path_2 = '/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h/'
#path_3 = '/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_16h15/'
#path_4 = '/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/20_02_2019_16h30/FAM/'
#path_folders = [path1 path2 path3]
path_5 = "/media/rblin/87c4f13b-ad62-44ef-babf-70c3e7c8a343/polar/13_05_2019_15h_I/"


#rename_frames_in_order_param(path_5)

path_rgb_movie = "/media/rblin/LaCie/Aquisitions_goPro_Polar/07_05_2019_16h30/RGB/frames/"
final_path = "/home/rblin/Documents/17_05_2019_16h30_rgb/"
rename_rgb(path_rgb_movie)

move_rgb(path_rgb_movie, final_path)