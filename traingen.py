# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:05:38 2020

@author: Nilesh
"""

import cv2
import numpy as np
import h5py
from PIL import Image
import skimage
import math

img_width = 416
img_height = 416
imgcv = np.zeros((416,416,3), np.uint8)
font = [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
size = 2
num_of_train_images = 500
num_of_test_images = 100
imageprop = {}

text = ['hi','there','i','am','nilesh','sutar','this','is','very','amazing',
        'cool','nice','being','happy','shit','nothing',
        'neat','discipline','rocks','booring','kinda','real','rocking',
        'sensible','kind','small','big','smart','object','goodboy','hello',
        'faster','generate','font','model','random','support','detection','zebra']

allfiles = []
pagetype = ['ruled1','ruled2','blank']


def drawBoundingBox(imgcv,label,font,fontsize,linethickness,position):
    for labeltext in label:
        labelSize = cv2.getTextSize(labeltext,font,fontsize,linethickness)
        xmin = position[0]
        ymin = position[1]
        xmax = xmin+labelSize[0][0]
        ymax = ymin+int(labelSize[0][1])    
        word_height = ymax - ymin                
         
        ymax = ymax + math.ceil(word_height/3.0)
        print("drawbox",xmin,ymin,xmax,ymax)   
        #cv2.rectangle(imgcv,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        cv2.putText(imgcv,labeltext,(xmin,ymax-math.ceil(word_height/3.0)),font,fontsize,(0,0,0),linethickness)
        #cv2.imshow('image',imgcv)                        
        #cv2.waitKey(0)         
        #cv2.destroyAllWindows()  
    return imgcv              


def generateData(folder = 'train'):
    global text,font, img_width, img_height,datafile, allfiles,testX,testY,trainX,trainY
    trainX = []
    trainY = []
    testX = []
    testY = []
    
    if folder == "train":
        # filepath = "C://Users//Nilesh//tensorflow//models//research//object_detection//images//train//"
        filepath = "images//train//"
        datafile = open('traindata.csv','w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+ 'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        datafile.write(datastring)
        num_of_images = num_of_train_images
    elif folder == "test":
        # filepath = "C://Users//Nilesh//tensorflow//models//research//object_detection//images//test//"
        filepath = "images//test//"
        datafile = open('testdata.csv','w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+ 'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        datafile.write(datastring)
        num_of_images = num_of_test_images
        
    for count in range(num_of_images):
        img_height = np.random.randint(300,600)
        img_width = np.random.randint(300,600)
        imgcv = np.zeros((img_height,img_width,3), np.uint8)
        fillcolor = np.random.randint(200,255)
        imgcv.fill(fillcolor)
        num_of_words = np.random.randint(1,100)
        labellist = []
        startpos = [np.random.randint(1,40),np.random.randint(20,40)]
        xmin = startpos[0]
        ymin = startpos[1]
        positionList = []
        newfilepath = filepath + str(count)+'.jpg'
        fontchoice = np.random.choice(font)
        fontsize = np.random.choice([0.5,0.75,1,1.25])
        linethickness = np.random.randint(1,3)
        
        for i in range(num_of_words):
            word = np.random.choice(text)
            wordSize = cv2.getTextSize(word,fontchoice,fontsize,linethickness)
            wordlength = wordSize[0][0]
            wordheight = int(wordSize[0][1])
            margin = np.random.randint(5,20)
            nymin = ymin - math.ceil(wordheight/3.0)
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
                positionList.append((xmin,nymin,xmax,ymax))
                                 
                xmin = xmax + margin
        
        indexList = list(np.arange(len(labellist)))
        randomIndexList = []
        if np.random.random() > 0.33 and len(indexList) > 50 :
            randomIndexList = np.random.choice(indexList,size = int(len(indexList)*0.5), replace = False)
        else:
            randomIndexList = indexList
            
        for index in randomIndexList:
            xmin = positionList[index][0]
            nymin = positionList[index][1]
            xmax = positionList[index][2]
            ymax = positionList[index][3]
            print("generate",xmin,nymin,xmax,ymax)
            imgcv = drawBoundingBox(imgcv,[labellist[index]],fontchoice,fontsize,linethickness,[xmin,nymin])
            datastring = newfilepath + ','+str(img_height)+','+str(img_width)+',' + 'text'+',' + str(xmin) + ',' + str(nymin) + ',' + str(xmax) + ',' + str(ymax) +'\n'
            datafile.write(datastring)
            
        if np.random.random() > 0.5:
            imgcv = skimage.util.random_noise(imgcv, mode="gaussian")
            imgcv = np.array(fillcolor*imgcv,dtype='uint8')
            imgcv = cv2.blur(imgcv,(3,3))
            
            
        if folder == 'train':
            trainX.append(imgcv)
            trainY.append(np.array([xmin,ymin,xmax,ymax,1]))
        else:
            testX.append(imgcv)
            testY.append([xmin,ymin,xmax,ymax,1])
            
        print(cv2.imwrite(filepath + str(count) + '.jpg', imgcv) )
        # print(labellist,positionList)
        
    if folder == 'train':
        return np.array(trainX),np.array(trainY)
    else:
        return np.array(testX), np.array(testY)

def store_many_hdf5(images):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File("test.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    # meta_set = file.create_dataset(
    #     "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    # )
    file.close()      

X,Y = generateData('train')

datafile.close()

                
            
            
            
        
        
    