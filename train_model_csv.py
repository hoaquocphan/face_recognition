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
import csv
import pandas as pd
import ntpath

from lib import create_alexnet_model
print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=48)
args = parser.parse_args()
num_class = args.n

train_csv = "train.csv"
valid_csv = "valid.csv"
test_csv = "test.csv"
DIR = "UTKFace"


with open(train_csv,newline='') as csvfile:
    train_data = [row for row in csv.DictReader(csvfile)]
with open(valid_csv,newline='') as csvfile:
    valid_data = [row for row in csv.DictReader(csvfile)]
with open(test_csv,newline='') as csvfile:
    test_data = [row for row in csv.DictReader(csvfile)]

# define function
def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

def decode_jpeg_and_label(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = tf.strings.split(tf.expand_dims(label.values[6], axis=-1), sep='_')
    label = label.values[0]
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int


database_train = pd.read_csv(train_csv)
database_valid = pd.read_csv(valid_csv)

def decode_jpeg_and_label_train(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    #########################################################
    # issue here: filename should be string, not tensor
    filename = ntpath.basename(filename)

    x = database_train[database_train.image_name == filename]
    label = x['label'].to_string(index=False)
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int

'''
print("decode_jpeg_and_label------------------------------")
image, label_int = decode_jpeg_and_label("/media/sf_ubuntu20/AI/face_recognition/train/3_10_1_0_2_20161219204012884.jpg.chip.jpg")


print("decode_jpeg_and_label_train------------------------")
image, label_int = decode_jpeg_and_label_train("/media/sf_ubuntu20/AI/face_recognition/UTKFace/10_0_0_20170110224500062.jpg.chip.jpg")
'''


def decode_jpeg_and_label_valid(filename):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    filename = ntpath.basename(filename)
    x = database_valid[database_valid.image_name == filename]
    label = x['label'].to_string(index=False)
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int

def decode_jpeg_and_label_test(id):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    database = pd.read_csv(test_csv)
    x = database[database.image_name == filename]
    label = x['label'].to_string(index=False)
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int

model=create_alexnet_model(num_class)
checkpoint_path = "model_alexnet/cp.ckpt"

train_image_path_ori = []
train_image_path = []
validation_image_path = []
test_image_path = []



for i in range(num_class):
    train_image_path_ori.append('train/'+ str(i) + '_*jpg')



for i in train_data:
    train_image_path.append(DIR + "/" + i['image_name'])
    #train_image_path.append(i['id'])

for i in valid_data:
    validation_image_path.append(DIR + "/" + i['image_name'])
    #validation_image_path.append(i['id'])

'''
for i in test_data:
    #test_image_path.append(DIR + "/" + i['image_name'])
    test_image_path.append(i['id'])
'''


filenames_traindataset_ori = tf.data.Dataset.list_files(train_image_path_ori, shuffle=True)
print("----------------")
print("filenames_traindataset_ori")
print(filenames_traindataset_ori)


#hangup
filenames_traindataset = tf.data.Dataset.list_files(train_image_path,  shuffle=True)
print("----------------")
print("filenames_traindataset")
print(filenames_traindataset)
print("----------------")

filenames_validdataset = tf.data.Dataset.from_tensor_slices(validation_image_path)
#filenames_testdataset = tf.data.Dataset.from_tensor_slices(test_image_path)


#filenames_traindataset = tf.data.Dataset.list_files(train_image_path, shuffle=True)
#filenames_validdataset = tf.data.Dataset.list_files(validation_image_path, shuffle=True)
#filenames_testdataset = tf.data.Dataset.list_files(test_image_path, shuffle=True)



#for element in filenames_traindataset:
#    print(element)

#print(validation_image_path)
#print(filenames_validdataset)
#print(test_image_path)
#print(filenames_testdataset)


image_traindataset_ori = filenames_traindataset_ori.map(decode_jpeg_and_label)
image_traindataset = filenames_traindataset.map(decode_jpeg_and_label_train)
image_validdataset = filenames_validdataset.map(decode_jpeg_and_label_valid)
#image_testdataset = filenames_testdataset.map(decode_jpeg_and_label_test)

#print(image_traindataset)
#print(image_testdataset)


'''
image_data = "validation/3_10_1_0_2_20161219140525218.jpg.chip.jpg"
image, label = decode_jpeg_and_label(image_data)
print(image)
print(label)
'''

'''
image_id = "8460"
image, label = decode_jpeg_and_label_train(image_id)
print(image)
print(label)
'''




#train_ds_size = tf.data.experimental.cardinality(image_traindataset).numpy()
#validation_ds_size = tf.data.experimental.cardinality(image_validdataset).numpy()

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
print("train model!")
model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(validation_ds)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))

