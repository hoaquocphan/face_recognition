# face_recognition

down load data: https://susanqq.github.io/UTKFace/

# prepare data:
python3 prepare_data.py --d UTKFace/

# train model:
python3 train_model.py --n <number_of_class>

# test model:
python3 test_model.py --n <number_of_class>

# Note:
1/ prepare_data.py script will create data with 48 classes, 
Update this script If you want to create data with different number of class
2/ number_of_class <=48, default is 48