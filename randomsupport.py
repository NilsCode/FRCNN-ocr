# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:50:23 2020

@author: Nilesh
"""

# split into train and test set
from os import listdir
import pandas as pd

train = pd.read_csv('traindata.csv',header = None)
train.columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] 
test = pd.read_csv('testdata.csv',header = None)
test.columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

python model_main.py --logtostderr --model_dir=F:\\Projects\\Tensorflow\\FasterRcnn\\faster_rcnn_inception_v2_coco_2018_01_28\\ --pipeline_config_path=F:\\Projects\\Tensorflow\\FasterRcnn\\

python model_main.py --logtostderr --model_dir=training\\ --pipeline_config_path=training\\faster_rcnn_inception_v2_pets.config

set PYTHONPATH=C:\Users\Nilesh\tensorflow\models\research;C:\Users\Nilesh\tensorflow\models\research\slim

python generate_tfrecord.py --csv_input=images/traindata.csv --output_path=train.record --img_path=images/train

python generate_tfrecord.py --csv_input=images/testdata.csv --output_path=test.record --img_path=images/test