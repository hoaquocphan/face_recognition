import tensorflow as tf
import cv2
import numpy as np
#import onnxruntime
#import mxnet
#import mxnet as mx
#from mxnet.gluon.data.vision import transforms
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import walk
import math
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='dir') 
parser.add_argument('--p', type=str, default=1) 
args = parser.parse_args()

def create_new_label(dirname):
    image_name=[]
    label_array=[]
    count=-1
    for (dirpath, dirnames, filenames) in walk(dirname):
        image_name.extend(filenames)
    for image in image_name:
        old_label=image.split("_")[0]
        if int(image.split("_")[0]) not in label_array:
            count+=1
            label_array.append(int(image.split("_")[0]))
        new_label=str(count)
        input_name= os.path.join(dirname,image)
        output_name= os.path.join(dirname, new_label + "_" + image)
        os.rename(input_name, output_name)

def main():
    image_name = []
    input_dir = args.d
    train_dir = "train"
    validation_dir = "validation"
    test_dir = "test"

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(validation_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    if os.path.isdir(train_dir):
        os.rmdir(train_dir)
    if os.path.isdir(validation_dir):
        os.rmdir(validation_dir)
    if os.path.isdir(test_dir):
        os.rmdir(test_dir)
    
    shutil.copytree(input_dir, train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    image_name = []
    for (dirpath, dirnames, filenames) in walk(train_dir):
        image_name.extend(filenames)
    for image in image_name:
        class_name = ""
        if int(image.split("_")[0]) >=1 and int(image.split("_")[0]) <=10:
            class_name += "1"
        elif int(image.split("_")[0]) >=11 and int(image.split("_")[0]) <=19:
            class_name += "2"
        elif int(image.split("_")[0]) >=20 and int(image.split("_")[0]) <=39:
            class_name += "3"
        elif int(image.split("_")[0]) >=40 and int(image.split("_")[0]) <=49:
            class_name += "4"
        elif int(image.split("_")[0]) >=50 and int(image.split("_")[0]) <=60:
            class_name += "5"
        elif int(image.split("_")[0]) >=61 and int(image.split("_")[0]) <=116:
            class_name += "6"

        if int(image.split("_")[1]) ==0:
            class_name += "1"
        elif int(image.split("_")[1]) ==1:
            class_name += "2"

        if int(image.split("_")[2]) ==0 or int(image.split("_")[2]) ==4:
            class_name += "1"
        elif int(image.split("_")[2]) ==1:
            class_name += "2"
        elif int(image.split("_")[2]) ==2:
            class_name += "3"
        elif int(image.split("_")[2]) ==3:
            class_name += "4"
        input_name= os.path.join(train_dir,image)
        output_name= os.path.join(train_dir,class_name + "_" + image)
        os.rename(input_name, output_name)
    
    image_name = []
    image_name_class = []
    cur_class = 0
    count = 1
    for (dirpath, dirnames, filenames) in walk(train_dir):
        image_name.extend(filenames)

    for image in image_name:
        if cur_class != int(image.split("_")[0]):
            if cur_class != 0:
                print("current class: ",cur_class)
                print("count: ",count)
                test_image = random.choice(image_name_class)
                input_name= os.path.join(train_dir,test_image)
                output_name= os.path.join(test_dir,test_image)
                os.rename(input_name, output_name)
                image_name_class.remove(test_image)
                for i in range(round(count/10)):
                    valid_image = random.choice(image_name_class)
                    input_name= os.path.join(train_dir,valid_image)
                    output_name= os.path.join(validation_dir,valid_image)
                    os.rename(input_name, output_name)
                    image_name_class.remove(valid_image)
                image_name_class = []
            cur_class = int(image.split("_")[0])
            count = 1
            image_name_class.append(image)
        else:
            count+=1
            image_name_class.append(image)
    
    print("current class: ",cur_class)
    print("count: ",count)
    test_image = random.choice(image_name_class)
    input_name= os.path.join(train_dir,test_image)
    output_name= os.path.join(test_dir,test_image)
    os.rename(input_name, output_name)
    image_name_class.remove(test_image)
    for i in range(round(count/10)):
        valid_image = random.choice(image_name_class)
        input_name= os.path.join(train_dir, valid_image)
        output_name= os.path.join(validation_dir,valid_image)
        os.rename(input_name, output_name)
        image_name_class.remove(valid_image)
    image_name_class = []
    
    create_new_label(train_dir)
    create_new_label(validation_dir)
    create_new_label(test_dir)



if __name__ == "__main__":
    main()