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

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label


if args.m == "normal":
    model=create_model(num_class)
    checkpoint_path = "model/cp.ckpt"
elif args.m =="alexnet":
    model=create_alexnet_model(num_class)
    checkpoint_path = "model_alexnet/cp.ckpt"

    train_labels=np.expand_dims(train_labels, axis=1)
    validation_labels=np.expand_dims(validation_labels, axis=1)
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))




checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)



# Train model by train image 
print("training model")

if args.m == "normal":
    model.fit(train_images, train_labels, epochs=50,validation_data=(validation_images, validation_labels), callbacks=[cp_callback])
elif args.m =="alexnet":
    model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))