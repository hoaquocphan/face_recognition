# face_recognition

down load data: https://susanqq.github.io/UTKFace/

## prepare data:

# prepare data to 3 parts: train, test, validation. 
# dir: create data folder: train, test, validation
# csv: create 3 csv file: train.csv, test.csv, validation.csv (ongoing)
python3 prepare_data.py --input UTKFace/ --option dir


## train model

python3 train_model.py --option <option>

# option: "normal" train model with preload data, "dataset" train model with dataset



# train model(load image and label from csv file)(need to update)

python3 train_model_csv.py

## test model:

python3 test_model.py --n <number_of_class> --m <model_name>

# Note:
1/ prepare_data.py script will create data with 48 classes, 
Update this script If you want to create data with different number of class

2/ number_of_class <=48, default is 48

3/ model_name is: normal or alexnet