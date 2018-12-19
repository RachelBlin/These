import os
import numpy as np
import imageio
import matplotlib.pyplot as plt

path_folder = "DOP_moyen/brouillard_dm"

files = os.listdir(path_folder)

print(files)

x_axis = [i for i in range(len(files))]
y_axis = np.zeros((len(files),1))

for i in range(len(files)):
    image_temp = imageio.imread(path_folder + "/" + files[i])
    mean = np.mean(image_temp)
    y_axis[i] = int(mean)

print(y_axis)

plt.figure(1)
plt.axhline(y=np.mean(y_axis), color='r', linestyle='-')
plt.plot(x_axis, y_axis, "b+")
plt.title("DOP moyen brouillard début de matinée")
plt.savefig("graphiques/DOP/brouillard_dm.png")
plt.show()

plt.figure(2)
plt.hist(y_axis, int(len(files)/5))
plt.savefig("graphiques/DOP/hist_brouillard_dm.png")
plt.show()
