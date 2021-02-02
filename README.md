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



## test model:

python3 test_model.py --option onnx --age 10

