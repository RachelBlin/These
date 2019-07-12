import os

def RetinaNet_eval(path_weights):
    #try:
    print(type(path_weights))
    weights = sorted(os.listdir(path_weights))
    #except FileNotFoudError:
        #print("No such file or directory ", path_weights)

    for weight in weights:
        os.system('python /home/rblin/Documents/keras-retinanet/keras_retinanet/bin/evaluate.py pascal /home/rblin/Documents/New_polarimetric_DB_3239 test_polar/PARAM_POLAR/Params test_polar/LABELS /media/rblin/LaCie/weights/Params/' + weight + ' --convert-model')

path_weights = "/media/rblin/LaCie/weights/Params/"
RetinaNet_eval(path_weights)