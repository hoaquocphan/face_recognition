from os import walk
import os
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
import cv2
import onnxruntime
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import get_label, get_images, create_model, create_alexnet_model
import argparse

###########################################
#this is test model file for train_model_dataset.py
#python3 train_model_dataset.py  
#python3 -m tf2onnx.convert --saved-model model_alexnet_savedmodel/ --output model_alexnet.onnx --inputs conv2d_input:0[1,227,227,3] --inputs-as-nchw conv2d_input:0
#python3 test_model_dataset.py
###########################################
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='car.jpg')
parser.add_argument('--age', type=int, default=5)
args = parser.parse_args()


test_dir = "test"

def preprocess(img):   
    image = cv2.imread(img)
    image = cv2.resize(image, (227, 227))
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = input_img.reshape(1, 3, 227, 227)
    return input_img

def softmax(x): 
    """Compute softmax values for each sets of scores in x.""" 
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0) 

def process_imagename(img):
    img = img.split("/")[-1]
    label = img.split("_")[0]
    class_name = img.split("_")[1]
    sex = class_name[-1]
    age = class_name[:-1]
    return int(label), int(sex), int(age)


def main(): 
    model_path = "model_alexnet_nchw.onnx"
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print('input_name: {}'.format(input_name))
    print('input_shape: {}'.format(input_shape))
    print('output_name: {}'.format(output_name))
    print('output_shape: {}'.format(output_shape))

    image_name = []
    data = []
    test_data = []
    for (dirpath, dirnames, filenames) in walk(test_dir):
        image_name.extend(filenames)
    for image in image_name:
        label, sex, age = process_imagename(image)
        data.append(label)
        data.append(sex)
        data.append(age)
        test_data.append(data)
        data = []

    right = 0
    wrong = 0
    for image in image_name:
        image = os.path.join(test_dir,image)
        input_img = preprocess(image)
        raw_result = session.run([], {input_name: input_img})
        #print(raw_result[0])
        output_data = np.array(raw_result[0]).squeeze(axis=0)
        out_score = softmax(output_data)
        #print("\n")
        #print('score: {}'.format(np.amax(out_score)))
        #print('detect items: {}'.format( np.where(out_score == np.amax(out_score))[0][0]))
        detect_label =  np.where(out_score == np.amax(out_score))[0][0]
        input_label, input_sex, input_age = process_imagename(image)
        
        detect_sex = -1
        detect_age = -1
        for data in test_data:
            if int(data[0]) == detect_label:
                detect_sex = int(data[1])
                detect_age = int(data[2])
                break
        
        if detect_sex != input_sex:
            wrong +=1
        elif abs(detect_age - input_age) >= args.age:
            wrong +=1
        else:
            right+=1

    print("accuracy: ", right/(right+wrong))

if __name__ == "__main__":
    main()


