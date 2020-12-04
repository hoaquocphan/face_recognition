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

print(tf.__version__)


train_labels = get_label("train")
validation_labels = get_label("validation")

train_images = get_images("train")
validation_images = get_images("validation")


num_class = 48

model=create_model(num_class)

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
# Train model by train image 
model.fit(train_images, train_labels, epochs=50,validation_data=(validation_images, validation_labels), callbacks=[cp_callback])



test_loss, test_acc = model.evaluate(validation_images,  validation_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
