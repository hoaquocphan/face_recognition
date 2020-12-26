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
model_name=args.m

test_labels = get_label("test",num_class)
test_images = get_images("test",num_class,model_name)


if args.m == "normal":
    model=create_model(num_class)
    checkpoint_path = "model/cp.ckpt"
elif args.m =="alexnet":
    model=create_alexnet_model(num_class)
    checkpoint_path = "model_alexnet/cp.ckpt"
    test_labels=np.expand_dims(test_labels, axis=1)



model.load_weights(checkpoint_path)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("model, accuracy: {:5.2f}%".format(100 * test_acc))
