from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(img):
    image = cv2.imread(img)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_img = cv2.resize(input_img, (182, 182)) 
    return input_img

def get_label(dir,num_class):
    image_name = []
    labels = []
    count = 0
    image_name_1 = []
    image_name_2 = []
    for (dirpath, dirnames, filenames) in walk(dir):
        image_name.extend(filenames)
    for image in image_name:
        if image[1] == "_":
            image_name_1.append(image)
        else:
            image_name_2.append(image)
    image_name_1.sort()
    image_name_2.sort()
    image_name = image_name_1+image_name_2
    for image in image_name:
        if image.split("_")[0] not in labels:
            count +=1
        if count > num_class:
            break
        labels.append(image.split("_")[0])
    labels = [int(numeric_string) for numeric_string in labels]
    labels=np.array(labels)
    return labels

def get_images(dir,num_class):
    image_name=[]
    images =[]
    image_name2=[]
    image_name_1 = []
    image_name_2 = []
    count = 0
    for (dirpath, dirnames, filenames) in walk(dir):
        image_name.extend(filenames)
    for image in image_name:
        if image[1] == "_":
            image_name_1.append(image)
        else:
            image_name_2.append(image)
    image_name_1.sort()
    image_name_2.sort()
    image_name = image_name_1+image_name_2
    for image in image_name:
        if image.split("_")[0] not in image_name2:
            count +=1
        if count > num_class:
            break
        image_name2.append(image.split("_")[0])
        image_file = dir + "/" + image
        image=preprocess_image(image_file)
        images.append(image)
    images = np.array(images)
    images = images/255.0
    return images


def create_model(num_class):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(182, 182)),
        tf.keras.layers.Dense(640, activation='relu'),
        tf.keras.layers.Dense(int(num_class)) #number of class
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()
    return model

def create_alexnet_model(num_class):
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(int(num_class), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    model.summary()
    return model