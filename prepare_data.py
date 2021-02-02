#import tensorflow as tf
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
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='dir') #dir: copy data to 3 directories: train, test, validation; csv: create 3 csv files: train.csv, test.csv, validation.csv
parser.add_argument('--input', type=str, default='dir') #input image dir
args = parser.parse_args()


train_dir = "train"
validation_dir = "validation"
test_dir = "test"
train_csv = "train.csv"
valid_csv = "valid.csv"
test_csv = "test.csv"
input_dir = args.input

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

def copy_test_valid_data(image_name_class,count):
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

def create_data_dir():
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(validation_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    if os.path.isdir(train_dir):
        os.rmdir(train_dir)
    if os.path.isdir(validation_dir):
        os.rmdir(validation_dir)
    if os.path.isdir(test_dir):
        os.rmdir(test_dir)
    
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
    os.mkdir(train_dir)
    image_name = []
    for (dirpath, dirnames, filenames) in walk(input_dir):
        image_name.extend(filenames)
    for image in image_name:
        if image.split("_")[2] == "2" and int(image.split("_")[0]) <= 100: #using data of asia and upto 100 year old
            path_file = os.path.join(input_dir,image)
            shutil.copy(path_file,train_dir)

    image_name = []
    for (dirpath, dirnames, filenames) in walk(train_dir):
        image_name.extend(filenames)
    for image in image_name:
        class_name = ""
        class_name += image.split("_")[0]
        class_name += image.split("_")[1]
        input_name= os.path.join(train_dir,image)
        output_name= os.path.join(train_dir,class_name + "_" + image)
        os.rename(input_name, output_name)
    create_new_label(train_dir)
    image_name_class = []
    cur_class = -1
    count = 1
    image_name = []
    for (dirpath, dirnames, filenames) in walk(train_dir):
        image_name.extend(filenames)

    for image in image_name:
        if cur_class != int(image.split("_")[0]):
            if cur_class != -1:
                copy_test_valid_data(image_name_class,count)
                image_name_class = []
            cur_class = int(image.split("_")[0])
            count = 1
            image_name_class.append(image)
        else:
            count+=1
            image_name_class.append(image)
    
    copy_test_valid_data(image_name_class,count)

def csv_writer(header, data, filename, option):
    with open (filename, "w", newline = "") as csvfile:
        if option == "write":
            movies = csv.writer(csvfile)
            movies.writerow(header)
            for x in data:
                movies.writerow(x)
        elif option == "update":
            writer = csv.DictWriter(csvfile, fieldnames = header)
            writer.writeheader()
            writer.writerows(data)
        else:
            print("Option is not known")

'''
def updater(filename):
    with open(filename, newline= "") as file:
        readData = [row for row in csv.DictReader(file)]
        readData[0]['image_name'] = '5.0'
    readHeader = readData[0].keys()
    csv_writer(readHeader, readData, filename, "update")
'''

def create_data(filename,input_dir):
    header = ["image_name", "class_name", "label"]
    image_name = []
    data=[]
    for (dirpath, dirnames, filenames) in walk(input_dir):
        image_name.extend(filenames)
    for i in image_name:
        temp=[]
        temp.append(i)
        data.append(temp)
    csv_writer(header, data, filename, "write")

def sort_csv(column_name,filename):
    with open(filename,newline='') as csvfile:
        readData = [row for row in csv.DictReader(csvfile)]
        sortedlist = sorted(readData, key=lambda row:(row[column_name]), reverse=False)
        readHeader = sortedlist[0].keys()
        csv_writer(readHeader, sortedlist, filename, "update")

def add_label(filename):
    with open(filename,newline='') as csvfile:
        readData = [row for row in csv.DictReader(csvfile)]
        for i in readData:
            class_name = ""
            
            if int(i['image_name'].split("_")[0]) >=1 and int(i['image_name'].split("_")[0]) <=10:
                class_name += "1"
            elif int(i['image_name'].split("_")[0]) >=11 and int(i['image_name'].split("_")[0]) <=19:
                class_name += "2"
            elif int(i['image_name'].split("_")[0]) >=20 and int(i['image_name'].split("_")[0]) <=39:
                class_name += "3"
            elif int(i['image_name'].split("_")[0]) >=40 and int(i['image_name'].split("_")[0]) <=49:
                class_name += "4"
            elif int(i['image_name'].split("_")[0]) >=50 and int(i['image_name'].split("_")[0]) <=60:
                class_name += "5"
            elif int(i['image_name'].split("_")[0]) >=61 and int(i['image_name'].split("_")[0]) <=116:
                class_name += "6"

            if int(i['image_name'].split("_")[1]) ==0:
                class_name += "1"
            elif int(i['image_name'].split("_")[1]) ==1:
                class_name += "2"

            if int(i['image_name'].split("_")[2]) ==0 or int(i['image_name'].split("_")[2]) ==4:
                class_name += "1"
            elif int(i['image_name'].split("_")[2]) ==1:
                class_name += "2"
            elif int(i['image_name'].split("_")[2]) ==2:
                class_name += "3"
            elif int(i['image_name'].split("_")[2]) ==3:
                class_name += "4"
            i['class_name'] = class_name

        readHeader = readData[0].keys()
        csv_writer(readHeader, readData, filename, "update")

    sort_csv("class_name",filename)
    with open(filename,newline='') as csvfile:
        readData = [row for row in csv.DictReader(csvfile)]
        count=-1
        label_array=[]
        for i in readData:
            if int(i['class_name']) not in label_array:
                count+=1
                label_array.append(int(i['class_name']))
            i['label'] = str(count)
        readHeader = readData[0].keys()
        csv_writer(readHeader, readData, filename, "update")


def create_data_csv():
    num_class = 48
    create_data(train_csv,input_dir)
    add_label(train_csv)

    with open(train_csv,newline='') as csvfile:
        readData = [row for row in csv.DictReader(csvfile)]
    train_data_all=[]
    test_data_all=[]
    valid_data_all=[]

    for n in range(num_class):
        train_data=[]
        test_data=[]
        valid_data=[]
        for i in readData:
            if i["label"] == str(n):
                train_data.append(i)
        num_image = len(train_data)
        random_data = random.choice(train_data)
        train_data.remove(random_data)
        test_data.append(random_data)
        
        for i in range(round(num_image/10)):
            random_data = random.choice(train_data)
            train_data.remove(random_data)
            valid_data.append(random_data)
        train_data_all.extend(train_data)
        valid_data_all.extend(valid_data)
        test_data_all.extend(test_data)

    readHeader = readData[0].keys()
    csv_writer(readHeader, test_data_all, test_csv, "update")
    csv_writer(readHeader, valid_data_all, valid_csv, "update")
    csv_writer(readHeader, train_data_all, train_csv, "update")

def main():

    if not os.path.isdir(input_dir):
        print("input directory doesn't exist")
    else:
        if args.option == "dir":
            create_data_dir()
        elif args.option == "csv":
            create_data_csv()
        else:
            print("currently, prepare_data.py doesn't support this option")
    

if __name__ == "__main__":
    main()