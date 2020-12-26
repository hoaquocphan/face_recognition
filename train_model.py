from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import get_label, get_images, create_model, create_alexnet_model
import argparse
print(tf.__version__)


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=48)
parser.add_argument('--m', type=str, default='normal')
args = parser.parse_args()

num_class=args.n


print("loading train label")
npy_file = "train_labels_" + str(num_class) + ".npy"
if os.path.isfile(npy_file):
    train_labels = np.load(npy_file) 
else:
    train_labels = get_label("train",num_class)
    np.save(npy_file, train_labels)

print("loading validation label")
npy_file = "validation_labels_" + str(num_class) + ".npy"
if os.path.isfile(npy_file):
    validation_labels = np.load(npy_file)
else:
    validation_labels = get_label("validation",num_class)
    np.save(npy_file, validation_labels)


print("loading train image")
npy_file = "train_images_" + str(num_class) + ".npy"
if os.path.isfile(npy_file):
    train_images = np.load(npy_file)
else:
    train_images = get_images("train",num_class)
    np.save(npy_file, train_images)

print("loading validation image")
npy_file = "validation_images_" + str(num_class) + ".npy"
if os.path.isfile(npy_file):
    validation_images = np.load(npy_file)
else:
    validation_images = get_images("validation",num_class)
    np.save(npy_file, validation_images)



if args.m == "normal":
    model=create_model(num_class)
    checkpoint_path = "model/cp.ckpt"
elif args.m =="alexnet":
    model=create_alexnet_model(num_class)
    checkpoint_path = "model_alexnet/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
# Train model by train image 
print("training model")
model.fit(train_images, train_labels, epochs=50,validation_data=(validation_images, validation_labels), callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
