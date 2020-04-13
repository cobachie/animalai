from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

image_root_dir = '/Volumes/Photos2016/animalai/'

classes = ["monkey", "boar", 'crow']
num_classes = len(classes)
image_size = 50

X = []
Y = []
for index, class_label in enumerate(classes):
  photos_dir = image_root_dir + class_label
  files = glob.glob(photos_dir + '/*.jpg')
  for i, file in enumerate(files):
    if i > 150: break
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X.append(data)
    Y.append(index)

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (x_train, x_test, y_train, y_test)
npy_path = image_root_dir + 'animal.npy'
np.save(npy_path, xy)
