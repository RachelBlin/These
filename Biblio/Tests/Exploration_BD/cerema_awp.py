import h5py
from PIL import Image
import matplotlib.pyplot as plt
#import magic

#magic.from_file('cerema0.25.hdf5')
# Ouverture du fichier
mon_fichier = h5py.File('/Users/rblin/Downloads/cerema0.25.hdf5', 'r')

# Affichage de la structure des dossiers dans le fichier hdf5
list_elmts = [key for key in mon_fichier['/'].keys()]
"""for key in list_elmts:
    print(key)
    print(type(mon_fichier['/'][key]))
    print(mon_fichier['/'][key])
    print([key for key in mon_fichier['/'][key].keys()])"""

# Accès aux dossiers et fichiers que l'on veut manipuler
mon_dataset_train = mon_fichier['train']

mon_dataset = mon_dataset_train['images']

# Afficher la première image de la BD CEREMA
for i in range(31):
    img = Image.fromarray(mon_dataset[i])
    img.save("/Users/rblin/Documents/img_cerema_test/"+str(i)+".jpg")

#imgplot = plt.imshow(img)
#plt.show()