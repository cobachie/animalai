from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

image_root_dir = '/Volumes/Photos2016/animalai/'

classes = ["monkey", "boar", 'crow']
num_classes = len(classes)
image_size = 50
num_test_data = 70

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, class_label in enumerate(classes):
  photos_dir = image_root_dir + class_label
  files = glob.glob(photos_dir + '/*.jpg')
  for i, file in enumerate(files):
    if i > 150: break
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)

    if i < num_test_data:
      X_test.append(data)
      Y_test.append(index)
    else:
      # -20度〜20度まで5度ずつ回転させる
      for angle in range(-20, 20, 5):
        img_r = image.rotate(angle)
        data = np.asarray(img_r)
        X_train.append(data)
        Y_train.append(index)

        # 反転
        img_t = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img_t)
        X_train.append(data)
        Y_train.append(index)

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)
y_train = np.asarray(Y_train)
y_test  = np.asarray(Y_test)

xy = (x_train, x_test, y_train, y_test)
npy_path = image_root_dir + 'animal_aug.npy'
np.save(npy_path, xy)
