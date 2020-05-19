# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:44:18 2020

@author: Nilesh
"""

from os import listdir
from numpy import asarray, zeros
from mrcnn.utils import Dataset
import pandas as pd
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import h5py
import numpy as np
import cv2

class TextConfig(Config):
    
	# define the name of the configuration
	NAME = "text_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 10
    
def drawBoundingBox(imgcv,label,font,position):
    for labeltext in label:
        labelSize=cv2.getTextSize(labeltext,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        xmin = position[0]
        ymin = position[1]
        xmax = xmin+labelSize[0][0]
        ymax = ymin+int(labelSize[0][1])
        #cv2.rectangle(imgcv,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv2.putText(imgcv,labeltext,(xmin,ymax),font,0.5,(0,255,0),1)
        #cv2.imshow('image',imgcv)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return imgcv

def generateData(folder = 'train'):
    img_width = 416
    img_height = 416
    imgcv = np.zeros((416,416,3), np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX
    size = 2
    num_of_train_images = 20
    num_of_test_images = 5
    
    text = ['hi','there','i','am','nilesh','sutar','this','is','very','amazing',
            'cool','nice','being','happy','shit','nothing',
            'neat','discipline','rocks','booring','kinda','real','rocking',
            'sensible','kind','small','big']
    trainX = []
    trainY = {}
    testX = []
    testY = {}
    
    if folder == "train":
        filepath = "F:\\Projects\\Tensorflow\\FasterRcnn\\train\\"
        # datafile = open('traindata.csv','w')
        num_of_images = num_of_train_images
    elif folder == "test":
        filepath = "F:\\Projects\\Tensorflow\\FasterRcnn\\test\\"
        # datafile = open('testdata.csv','w')
        num_of_images = num_of_test_images
        
    for count in range(num_of_images):
        
        imgcv = np.zeros((416,416,3), np.uint8)
        num_of_words = np.random.randint(1,150)
        labellist = []
        startpos = [np.random.randint(1,40),np.random.randint(1,40)]
        xmin = startpos[0]
        ymin = startpos[1]
        positionList = []
        newfilepath = filepath + str(count)+'.jpg'
        
        for i in range(num_of_words):
            word = np.random.choice(text)
            wordSize = cv2.getTextSize(word,font,0.5,2)
            wordlength = wordSize[0][0]
            wordheight = int(wordSize[0][1])
            margin = np.random.randint(5,20)
            xmax = xmin + wordlength
            ymax = ymin + wordheight
            # print(i)
            if ymax >= img_height:
                break
            
            elif xmax >= img_width:
                xmin = startpos[0]
                ymin += wordheight + margin
                
            else:
                labellist.append(word)
                positionList.append((xmin,xmax,ymin,ymax))
                imgcv = drawBoundingBox(imgcv,[word],font,[xmin,ymin])
                # datastring = newfilepath + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +',' + '0' + '\n'
                # datafile.write(datastring)
                
            if folder == 'train':
                if count not in trainY:
                    trainY[count] = []
                trainY[count].append([xmin,ymin,xmax,ymax])
            else:
                if count not in testY:
                    testY[count] = []
                testY[count].append(np.array([xmin,ymin,xmax,ymax]))
            xmin = xmax + margin
        if folder == 'train':
            trainX.append(imgcv)
            
        else:
            testX.append(imgcv)
    
        # print(cv2.imwrite(folder + '\\' + str(count) + '.jpg', imgcv) )
        # print(labellist,positionList)
        
    if folder == 'train':
        return np.array(trainX),trainY
    else:
        return np.array(testX), testY
    
    
def startTraining():
# prepare config
    config = TextConfig()
    config.BATCH_SIZE = 1
    config.display()
    X,Y = generateData('train')
    # define the model
    model = MaskRCNN(mode='training', model_dir='F:\\Projects\\Tensorflow\\FasterRcnn\\', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(X,Y, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

startTraining()