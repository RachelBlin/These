import os
from shutil import copyfile

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

path_folder = "/home/rblin/Documents/Aquisitions/Polar/Hiver/Ensoleille/FAM"
path_directory = "/home/rblin/Documents/BD_QCAV/train/POLAR"
first_frame = 113
last_frame = 134829
step = 25

new_frames = get_final_frames(path_folder, path_directory, first_frame, last_frame, step)
print(new_frames[3351])
