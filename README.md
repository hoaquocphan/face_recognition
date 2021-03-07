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


## example run options:

# 1/ prepare data to 3 directories, train model by dataset.

python3 prepare_data.py --input UTKFace/ --option dir

python3 train_model.py --option dataset

python3 -m tf2onnx.convert --saved-model model_alexnet_savedmodel/ --output model_alexnet_nchw.onnx --inputs conv2d_input:0[1,227,227,3] --inputs-as-nchw conv2d_input:0

python3 test_model.py --option onnx --age 10 --model_path model_alexnet_nchw.onnx

# 2/

python3 prepare_data.py --input UTKFace/ --option csv

