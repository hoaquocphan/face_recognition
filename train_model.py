from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import get_label, get_images, create_model
import argparse
print(tf.__version__)


num_class=48


print("loading train label")
if os.path.isfile("train_labels.npy"):
    train_labels = np.load('train_labels.npy') 
else:
    train_labels = get_label("train",num_class)
    np.save('train_labels.npy', train_labels)

print("loading validation label")
if os.path.isfile("validation_labels.npy"):
    validation_labels = np.load('validation_labels.npy')
else:
    validation_labels = get_label("validation",num_class)
    np.save('validation_labels.npy', validation_labels)

print("loading train image")
if os.path.isfile("train_images.npy"):
    train_images = np.load('train_images.npy')
else:
    train_images = get_images("train",num_class)
    np.save('train_images.npy', train_images)

print("loading validation image")
if os.path.isfile("validation_images.npy"):
    validation_images = np.load('validation_images.npy')
else:
    validation_images = get_images("validation",num_class)
    np.save('validation_images.npy', validation_images)




model=create_model(num_class)


checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
# Train model by train image 
print("training model")
model.fit(train_images, train_labels, epochs=50,validation_data=(validation_images, validation_labels), callbacks=[cp_callback])



test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
