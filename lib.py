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

