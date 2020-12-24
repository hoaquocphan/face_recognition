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

num_class = 48
test_labels = get_label("test",num_class)
test_images = get_images("test",num_class)



model=create_model(num_class)

checkpoint_path = "model/cp.ckpt"

model.load_weights(checkpoint_path)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
