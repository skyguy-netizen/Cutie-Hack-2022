# import cv2 as cv
# from tkinter import *
# import os
# from keras_preprocessing.image import ImageDataGenerator
# from keras.applications import MobileNetV2
# from keras.layers import AveragePooling2D
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers import Dense
# from keras.layers import Input
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.applications.mobilenet_v2 import preprocess_input
# from keras_preprocessing.image import img_to_array
# from keras_preprocessing.image import load_img
# from keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from imutils import paths
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime

# imagePaths = list(paths.list_images('/content/drive/My Drive/dataset'))
# data = []
# labels = []
# # loop over the image paths
# for imagePath in imagePaths:
#     # extract the class label from the filename
#     label = imagePath.split(os.path.sep)[-2]
#     # load the input image (224x224) and preprocess it
#     image = load_img(imagePath, target_size=(224, 224))
#     image = img_to_array(image)
#     image = preprocess_input(image)
#     # update the data and labels lists, respectively
#     data.append(image)
#     labels.append(label)
# # convert the data and labels to NumPy arrays
# data = np.array(data, dtype="float32")
# print(data)
# labels = np.array(labels)

# baseModel = MobileNetV2(weights="imagenet", include_top=False,
# input_shape=(224, 224, 3))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)
# model = Model(inputs=baseModel.input, outputs=headModel)
# for layer in baseModel.layers:
#     layer.trainable = False


# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)
# # partition the data into training and testing splits using 80% of
# # the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.20, stratify=labels, random_state=42)
# # construct the training image generator for data augmentation
# aug = ImageDataGenerator(
# 	rotation_range=20,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")

# INIT_LR = 1e-4
# EPOCHS = 20
# BS = 32
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])
# # train the head of the network
# print("[INFO] training head...")
# H = model.fit(
# 	aug.flow(trainX, trainY, batch_size=BS),
# 	steps_per_epoch=len(trainX) // BS,
# 	validation_data=(testX, testY),
# 	validation_steps=len(testX) // BS,
# 	epochs=EPOCHS)

# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")

# model.save('mask_recog_ver2.h5')

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np

model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "./Images/Dataset/train"
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))
VALIDATION_DIR = "./Images/Dataset/test"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))

checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history = model.fit_generator(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])