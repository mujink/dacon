import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
import tensorflow as tf
import scipy.signal as signal
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
#########데이터 로드

TRAIN_DIR = "LPD_competition/train"
IMG_SIZE = (224,224)

categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = np.zeros(nb_classes, dtype=np.float)
    label[idx] = 1

    uniform_distribution = np.full(nb_classes, 1 / nb_classes)
    # (0.001)
    deta = 0.01
    # 1 : 0.99
    # 0 : 0.0001
    smooth_onehot = label * (1 - deta) + deta * uniform_distribution

    image_dir = TRAIN_DIR + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        data = np.asarray(img)

        X.append(data)
        y.append(smooth_onehot)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("np/train_X.npy", arr=X)
np.save("np/train_y.npy", arr=y)
# x_pred = np.load("../data/npy/P_project_test.npy",allow_pickle=True)
x = np.load("np/train_x.npy",allow_pickle=True)
y = np.load("np/train_y.npy",allow_pickle=True)

print(x.shape)
print(y.shape)