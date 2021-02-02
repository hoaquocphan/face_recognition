from os import walk
import os
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import onnxruntime
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import get_label, get_images, create_model, create_alexnet_model
import argparse
print(tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='onnx')
parser.add_argument('--age', type=int, default=5)
args = parser.parse_args()

test_dir = "test"

def test_model_checkpoint(model_name, num_class):
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

def test_model_onnx(model_path):
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


def main():
    if args.option == 'checkpoint':
        num_class = 48
        model_name = 'normal'
        test_model_preload_data(model_name, num_class)
    elif args.option == 'onnx':
        model_path = "model_alexnet_nchw.onnx"
        test_model_onnx(model_path)
    else:
        print("currently, test_model.py doesn't support this test type")


if __name__ == "__main__":
    main()
