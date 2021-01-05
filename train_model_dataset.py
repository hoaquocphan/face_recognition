from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import argparse
from lib import create_alexnet_model
print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=48)
args = parser.parse_args()
num_class = args.n

# define function
def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

def decode_jpeg_and_label(filename):
  bits = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(bits)
  label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
  label = tf.strings.split(tf.expand_dims(label.values[1], axis=-1), sep='_')
  label = label.values[0]
  label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
  return image, label_int

model=create_alexnet_model(num_class)
checkpoint_path = "model_alexnet/cp.ckpt"

train_image_path = []
validation_image_path = []
for i in range(num_class):
    train_image_path.append('train/'+ str(i) + '_*jpg')
    validation_image_path.append('validation/'+ str(i) + '_*jpg')
filenames_traindataset = tf.data.Dataset.list_files(train_image_path, shuffle=True)
filenames_validdataset = tf.data.Dataset.list_files(validation_image_path, shuffle=True)

image_traindataset = filenames_traindataset.map(decode_jpeg_and_label)
image_validdataset = filenames_validdataset.map(decode_jpeg_and_label)
train_ds_size = tf.data.experimental.cardinality(image_traindataset).numpy()
validation_ds_size = tf.data.experimental.cardinality(image_validdataset).numpy()

train_ds = (image_traindataset
            .map(process_images)
#            .shuffle(buffer_size=train_ds_size)
            .batch(batch_size=32, drop_remainder=True))
validation_ds = (image_validdataset
            .map(process_images)
#            .shuffle(buffer_size=validation_ds_size)
            .batch(batch_size=32, drop_remainder=True))


checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(validation_ds)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))