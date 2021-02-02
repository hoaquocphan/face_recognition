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
parser.add_argument('--option', type=str, default='dataset')
args = parser.parse_args()

train_type = args.option 

train_dir = "train"
validation_dir = "validation"

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

def train_model_with_preload_data(model_name,num_class):
    print("-------------------------------")
    print("loading train label")
    train_labels = get_label("train",num_class)
    print('train_labels.shape: {}'.format(train_labels.shape))

    print("-------------------------------")
    print("loading validation label")
    validation_labels = get_label("validation",num_class)
    print('validation_labels.shape: {}'.format(validation_labels.shape))

    print("-------------------------------")
    print("loading train image")
    train_images = get_images("train",num_class,model_name)
    print('train_images.shape: {}'.format(train_images.shape))

    print("-------------------------------")
    print("loading validation image")
    validation_images = get_images("validation",num_class,model_name)
    print('validation_images.shape: {}'.format(validation_images.shape))

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

    if model_name == "normal":
        model.fit(train_images, train_labels, epochs=50,validation_data=(validation_images, validation_labels), callbacks=[cp_callback])
    elif model_name =="alexnet":
        model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[cp_callback])

    test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
    print("model, accuracy: {:5.2f}%".format(100 * test_acc))

def train_model_with_dataset(num_class):
    model=create_alexnet_model(num_class)
    checkpoint_path = "model_alexnet/cp.ckpt"

    train_image_path = []
    validation_image_path = []
    for i in range(num_class):
        train_image_path.append(train_dir + '/' + str(i) + '_*jpg')
        validation_image_path.append(validation_dir + '/'+ str(i) + '_*jpg')
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
    model.save("model_alexnet_savedmodel")
    test_loss, test_acc = model.evaluate(validation_ds)
    print("model, accuracy: {:5.2f}%".format(100 * test_acc))

def main():
    if train_type == "normal":
        num_class = 48
        model_name = "normal" # normal or alexnet
        train_model_with_preload_data(model_name,num_class)
    elif train_type == "dataset":
        num_class = 176
        train_model_with_dataset(num_class)
    else:
        print("currently, train_model.py doesn't support this train type")



if __name__ == "__main__":
    main()