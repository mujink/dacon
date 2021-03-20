
import numpy as np
import os

####
TEST_DIR = 'C:/data/LPD_competition/test'
TEST_SIZE = 72000
DIMENSION = 128
####

# prepare empty arrays for storing
test_files = os.listdir(TEST_DIR)
test_x = np.empty((TEST_SIZE, DIMENSION, DIMENSION, 3))
#==== until here, only dir ====#

# access to image files and make it into a numpy array (X, y)
for i, f in enumerate(test_files):
    img = image.load_img(os.path.join(TEST_DIR, f), target_size=(DIMENSION, DIMENSION)) # access to folder and load as image
    x = image.img_to_array(img) # now dtype is array
    x = np.expand_dims(x, axis=0) # MobileNet 전처리 위해서 4D tensor로 일단 만들어준다
    x = preprocess_input(x)
    x = np.squeeze(x) # 전처리했으니 다시 3D tensor로 만들어준다
    test_x[i,:,:,:] = x

print(test_x)
print(test_x.shape)