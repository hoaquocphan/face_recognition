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
    print(filename)
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    print("---------------")
    print(label)
    label = tf.strings.split(tf.expand_dims(label.values[1], axis=-1), sep='_')
    print(label)
    label = label.values[0]
    print(label)
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    print(label_int)
    print("---------------")
    return image, label_int

def decode_jpeg_and_label_train(id):
    print(id)
    print("hoaphan")
    id2 = tf.strings.split(tf.expand_dims(id, axis=-1))
    print(id2)
    #print(train_data[int(id)]['image_name'])
    return 1, 2
    '''
    bits = tf.io.read_file(DIR + "/" + train_data[int(id)]['image_name'])
    image = tf.image.decode_jpeg(bits)
    label = train_data[int(id)]['label']
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int
    '''

def decode_jpeg_and_label_valid(id):
    bits = tf.io.read_file(DIR + "/" + valid_data[int(id)]['image_name'])
    image = tf.image.decode_jpeg(bits)
    label = valid_data[int(id)]['label']
    label_int = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return image, label_int

def decode_jpeg_and_label_test(id):
    bits = tf.io.read_file(DIR + "/" + test_data[int(id)]['image_name'])
    image = tf.image.decode_jpeg(bits)
    label = test_data[int(id)]['label']
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
    #train_image_path.append(DIR + "/" + i['image_name'])
    train_image_path.append(i['id'])
for i in valid_data:
    #validation_image_path.append(DIR + "/" + i['image_name'])
    validation_image_path.append(i['id'])
for i in test_data:
    #test_image_path.append(DIR + "/" + i['image_name'])
    test_image_path.append(i['id'])

filenames_traindataset = tf.data.Dataset.from_tensor_slices(train_image_path)
filenames_validdataset = tf.data.Dataset.from_tensor_slices(validation_image_path)
filenames_testdataset = tf.data.Dataset.from_tensor_slices(test_image_path)

filenames_traindataset_ori = tf.data.Dataset.list_files(train_image_path_ori, shuffle=True)

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
#image_validdataset = filenames_validdataset.map(decode_jpeg_and_label_valid)
#image_testdataset = filenames_testdataset.map(decode_jpeg_and_label_test)

#print(image_traindataset)
#print(image_testdataset)


'''
image_data = "validation/2_113_5_0_2_20161219194354209.jpg.chip.jpg"
image, label = decode_jpeg_and_label(image_data)
print(image)
print(label)



image_id = "8460"
image, label = decode_jpeg_and_label_train(image_id)
print(image)
print(label)
'''



'''
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

'''