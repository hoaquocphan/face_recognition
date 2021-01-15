from os import walk
import os
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import operator
import random


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=48)
args = parser.parse_args()
num_class = args.n

DIR="UTKFace"
train_csv = "train.csv"
valid_csv = "valid.csv"
test_csv = "test.csv"

def writer(header, data, filename, option):
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

def updater(filename):
    with open(filename, newline= "") as file:
        readData = [row for row in csv.DictReader(file)]
        readData[0]['image_name'] = '5.0'
    readHeader = readData[0].keys()
    writer(readHeader, readData, filename, "update")

def create_data(filename):
    header = ["image_name", "class_name", "label"]
    image_name = []
    data=[]
    for (dirpath, dirnames, filenames) in walk(DIR):
        image_name.extend(filenames)
    for i in image_name:
        temp=[]
        temp.append(i)
        data.append(temp)
    writer(header, data, filename, "write")

def sort_csv(column_name,filename):
    with open(filename,newline='') as csvfile:
        readData = [row for row in csv.DictReader(csvfile)]
        sortedlist = sorted(readData, key=lambda row:(row[column_name]), reverse=False)
        readHeader = sortedlist[0].keys()
        writer(readHeader, sortedlist, filename, "update")

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
        writer(readHeader, readData, filename, "update")

    sort_csv("class_name",train_csv)
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
        writer(readHeader, readData, filename, "update")


def main():
    create_data(train_csv)
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
    writer(readHeader, test_data_all, test_csv, "update")
    writer(readHeader, valid_data_all, valid_csv, "update")
    writer(readHeader, train_data_all, train_csv, "update")
    

if __name__=="__main__":
    main()
