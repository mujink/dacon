# - 다중 클래스 이미지 분류를 금방 개발 가능하도록 템플릿을 작성

# - 클래스 수, 특징 추출 모델, 입력층 등을 쉽게 커스텀 및 업그레이드 가능하도록 하고, 고도화.

# - 사용자 입장에서는 사진 폴더별로 이미지만 나눠서 넣으면 쉽게 학습 가능.

# - 여기선 백본을 resnet v2 50으로 둠.

 

# 0. 라이브러리

# !pip install -q -U tf-hub-nightly
# 설치 및 임포트

import matplotlib.pylab as plt
import tensorflow as tf
import os
import numpy as np
import tensorflow_hub as hub
import pandas as pd
# from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 1. 데이터 준비

IMG_SIZE = (224,224)
BATCH_SIZE = 32

TEST_DIR = "LPD_competition/test/tests"
x = np.load("np/train_x.npy",allow_pickle=True)
y = np.load("np/train_y.npy",allow_pickle=True)
print(y.shape)
y = np.argmax(y, axis=-1)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)

print(y_train.shape)

# train_datagen = ImageDataGenerator(
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator()

# train_set = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
# valid_set = train_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)

# print(len(train_set))
# print(len(valid_set))

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


# AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
# valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)

# 먼저 이미지 분석에 있어서 없어선 안될 데이터 증강을 레이어로 만들어서 추후 모델에 포함시킬것임.

# 각 순전파마다 자동으로 데이터 증강이 되며, 매 학습마다 다른 이미지로 학습하는 결과를 만들어줌.(실전에서는 모델에 포함시키는게 아니라 학습 프로세스를 변경하여 학습시에만 이를 순전파시켜 이미지를 증강시킬것임.)

 
minmax_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)
# 데이터 스케일링 레이어.

# 이미지 데이터 값의 스케일을 0~255로 할건지 0~1로 할건지를 정함.

# -1~1의 경우는
Standard_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)
 
# 3. 특징 추출 모델 가져오기

# Create the base model from the pre-trained model Resnet v2 50
feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

IMG_SHAPE = IMG_SIZE + (3,)
base_model = hub.KerasLayer(feature_extractor_url,
                                         input_shape=IMG_SHAPE)
# 텐서플로 허브를 이용해서 CNN 특징추출 모델을 가져온 것.

# 텐서플로 허브를 사용해서 위와 같이 URL만 바꿔줘도 되는데,

# output은 확인할것.

# 위의 모델은 확인해보면 특징 추출 출력값이

# (32, 2048)

# 인데, 배치인 32를 제외하면 1x2048임.

# 즉, 만약 (32, 5, 5, 1280)같은 2d 필터 결과가 나오면, 이에 대해 

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# 라는 글로벌 에버리지 풀링 레이어를 CNN 뒤에 붙여줘서, (32, 5, 5, 1280)를 (32, 1280)으로 만들어줘야함.

# 일단 처음 학습시에는 잘 학습된 CNN은 동결해야하니
base_model.trainable = False
 
# 4. 모델 만들기

# 이제 조립만 남음.
prediction_layer = tf.keras.layers.Dense(1000)
# 이건 이미지 분류이니, 특징 추출맵에서 나온 것을 받아들여 몇개의 클래스로 반환하는 FC레이어를 하나 만듬.
# next_layer는 일단 분류를 위해 xor문제 해결 퍼셉트론에 따라 만들었는데 노드 개수를 정확히 모르겠음. 일단 노멀하게 사용되는 128을 사용.
# 그리고 prediction레이어는 정답데이터가 0~1의 확률값이 아닌 클래스 인덱스라서 소프트맥스는 사용 안함
"""
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = minmax_layer(x)
# x = Standard_layer(x)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
"""

from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPooling2D,AveragePooling2D,Dense,GlobalAveragePooling2D

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = minmax_layer(x)
# x = Standard_layer(x)
x = Conv2D(filters=128, kernel_size=(2,2),padding='same',activation='swish')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128, kernel_size=(2,2),padding='same',activation='swish')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=(2,2),padding='same',activation='swish')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
x = Conv2D(filters=64, kernel_size=(2,2),padding='same',activation='swish')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
x = GlobalAveragePooling2D()(x)
# x = base_model(x, training=False)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
# 모두 조립하면 먼저 IMG_SHAPE만큼 입력층을 가지고, 이걸 증강레이어에 넣고, 이걸 리스케일링 레이어에 넣음.
# 여기까진 입력 형태가 변화 없음.
# 이후 특징추출레이어에 넣어주고, 여기서 (batch, 2048)의 데이터가 나올텐데, 이를 flatten할필요 없이 Dropout에 넣어주고, 최종 결정인 5클래스 출력레이어를 끝으로 하고 이 그래프를 모델로 만들어줌.

model.summary() # 로 구조를 대략 파악해도 됨

# 5. 학습
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
# 컴파일 후 학습
es = EarlyStopping(monitor='val_acc', patience=10, mode='max')
cp = ModelCheckpoint("lotte2.h5", monitor='val_acc', save_best_only=True, mode='max')
lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.5, verbose=1)
initial_epochs = 100

history = model.fit(x_train, y_train,
                    callbacks=[cp,es,lr],
                    validation_data=(x_val,y_val),
                    epochs=initial_epochs
                  )

loss, acc = model.evaluate(x_val,y_val)
print("acc : ", acc)
model.save("my_model1")

#  6. 시각화

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training acc')
plt.plot(val_acc, label='Validation acc')
plt.legend(loc='lower right')
plt.ylabel('acc')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation acc')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
 

# ================== predict ==================
result = pd.read_csv("LPD_competition/sample.csv")

a = pd.DataFrame()
b = []
from tensorflow.keras.models import load_model
model= load_model("my_model1")

for _, _, filenames in os.walk(TEST_DIR):
    # Loop over the files
    for filename in range(len(filenames)):
        # Get filepath
        print(filename)
        filepath = os.path.join(TEST_DIR, str(filename) + '.jpg')
        # Get predictions
        raw_img = tf.keras.preprocessing.image.load_img(filepath, target_size=IMG_SIZE)  # Load the image
        img_array = tf.keras.preprocessing.image.img_to_array(raw_img)  # Convert to numpy array
        img_array = tf.expand_dims(img_array, 0)  # Reshaping - create batch axis
        predictions = model.predict(img_array)  # Make predictions
        b.append(np.argmax(predictions,axis=-1)[0])

    predict = pd.Series(b)  # Get the color with max probabilities


predict = pd.concat([a,predict],axis=1)
result.iloc[:,1] = predict.sort_index().values
result.to_csv('sample_resnetv250_1.csv')


# # 7. 파인튜닝
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
# cp = ModelCheckpoint("lotte2.h5", monitor='val_loss', save_best_only=True, mode='min')
# lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.5, verbose=1)

# base_model.trainable = True
# fine_tune_epochs = 100
# total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_set,
#                          callbacks=[cp,es,lr],
#                          epochs=total_epochs,
#                          initial_epoch=history.epoch[-1],
#                          validation_data=valid_set)
# model.save("my_model2")


# acc += history_fine.history['acc']
# val_acc += history_fine.history['val_acc']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training acc')
# plt.plot(val_acc, label='Validation acc')
# plt.ylim([0.8, 1])
# plt.plot([initial_epochs-1,initial_epochs-1],
#           plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='lower right')
# plt.title('Training and Validation acc')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.0])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# # ================== predict ==================
# result = pd.read_csv("LPD_competition/sample.csv")

# a = pd.DataFrame()
# b = []


# for _, _, filenames in os.walk(TEST_DIR):
#     # Loop over the files
#     for filename in range(len(filenames)):
#         # Get filepath
#         print(filename)
#         filepath = os.path.join(TEST_DIR, str(filename) + '.jpg')
#         # Get predictions
#         raw_img = tf.keras.preprocessing.image.load_img(filepath, target_size=IMG_SIZE)  # Load the image
#         img_array = tf.keras.preprocessing.image.img_to_array(raw_img)  # Convert to numpy array
#         img_array = tf.expand_dims(img_array, 0)  # Reshaping - create batch axis
#         predictions = model.predict(img_array)  # Make predictions
#         b.append(np.argmax(predictions,axis=-1)[0])

#     predict = pd.Series(b)  # Get the color with max probabilities


# predict = pd.concat([a,predict],axis=1)
# result.iloc[:,1] = predict.sort_index().values
# result.to_csv('sample_resnetv250_2.csv')