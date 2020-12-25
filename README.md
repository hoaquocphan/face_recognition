# face_recognition

down load data: https://susanqq.github.io/UTKFace/

# prepare data:
python3 prepare_data.py --d UTKFace/

# train model:
python3 train_model.py --n <number_of_class>

# test model:
python3 test_model.py --n <number_of_class>

number_of_class <=48