# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:30:27 2020

@author: Nilesh
"""

#! /usr/bin/env python
# -*- coding=utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
import sys
import time
import cv2
import pandas as pd

# Add object_detection to system path
OBJECT_DETECTION_PATH = r'C:\Users\Nilesh\tensorflow\models\research\object_detection'
sys.path.append(r"C:\Users\Nilesh\tensorflow\models\research")
sys.path.append(OBJECT_DETECTION_PATH)


# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
df = pd.read_csv(r'images\testdata.csv',index_col=False)

class DetectImage(object):
    category_index = 'index'
    sess = 'sess'

    # graph input and output
    image_tensor = 'Tensor'
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = 'Tensor'
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = 'Tensor'
    detection_classes = 'Tensor'
    num_detections = 'Tensor'

    def __init__(self, PATH_TO_CKPT='.pb', PATH_TO_LABELS='.pbtxt', NUM_CLASSES=-1):
        '''
        Load category_index, load graph, run sess 
        :param PATH_TO_CKPT: 
            Path to frozen detection graph. This is the actual model that is used for the object detection.
        :param PATH_TO_LABELS: 
            List of the strings that is used to add correct label for each box.
        :param NUM_CLASSES: 
            Number of class for model to detect 
        '''
        if not os.path.exists(PATH_TO_CKPT):
            print('PATH_TO_CKPT not exist')
            return
        if not os.path.exists(PATH_TO_LABELS):
            print('PATH_TO_LABELS not exist')
            return

        # Set category_index
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        # Load a (frozen) Tensorflow model into memory.
        print('Load graph')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # Open graph and sess
        self.sess = tf.Session(graph=detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def run_detect(self, image_np):
        '''
        run detect on a image
        :param image_np: image to detect
        :return: image with result, detection_boxes, detection_scores, detection_classes, num_detections
        '''

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        imgh = image_np.shape[0]
        imgw = image_np.shape[1]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Actual detection.
        time_begin = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        time_end = time.time()
        nbox = list(np.squeeze(boxes))
        scorelist = list(np.squeeze(scores))
        stBXlist = []
        for i in range(len(nbox)):
            ymin,xmin,ymax,xmax = list(nbox[i])
            ymin = int(ymin * imgh)
            xmin = int(xmin * imgw)
            ymax = int(ymax * imgh)
            xmax = int(xmax * imgw)
            if scorelist[i] > 0.5:
                stBXlist.append((xmin,ymin,xmax,ymax))
                cv2.rectangle(image_np, (xmin,ymin), (xmax,ymax),(0,0,255),1)
        sortedlist = sorted(stBXlist)
        #print("Boxes: ",list(box))
        #cv2.imshow('image',image_np)                        
        #cv2.waitKey(0)         
        #cv2.destroyAllWindows()
        self.get_avg_gpu_time(time_end - time_begin)
        
        return sortedlist

    frame_count = 50
    time_sum = 0
    def get_avg_gpu_time(self, dt):
        if self.frame_count == 50:
            print('average gpu time is: {}'.format(self.time_sum / 50))
            self.frame_count = 0
            self.time_sum = 0
        else:
            self.time_sum += dt
            self.frame_count += 1


if __name__ == '__main__':
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'C:\\Users\\Nilesh\\tensorflow\\models\\research\\object_detection\\inference_graph\\frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'training/labelmap.pbtxt')
    NUM_CLASSES = 1

    # Create DetectImage class
    di = DetectImage(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    PATH_TO_TEST_IMAGES_DIR = os.path.join(r'images//test//')
    TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR+'{}.jpg'.format(i) for i in range(0,100)]
    datafile = open(r'images\imageboxdata.csv','w')
    import skimage.io
    i = 0
    for image_path in TEST_IMAGE_PATHS:
        wordlist = list(df[df.filename == image_path].word[0])
        image_np = skimage.io.imread(image_path)
        boxes = di.run_detect(image_np)
        print(len(boxes))
        j = 0
        for box in boxes:
            datastring = str(i) +".jpg"+","+wordlist[j]+','+str(box[0])+","+str(box[1])+","+str(box[2])+","+str(box[3])+'\n'
            datafile.write(datastring)
            j += 1
        i += 1
    datafile.close()