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
font = [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
size = 2
num_of_train_images = 800
num_of_test_images = 100

imagewordprop = []

alphabets = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
             'q','r','s','t','u','v','w','x','y','z'}

ntext = []

text = ['hi','there','i','am','nilesh','sutar','this','is','very','amazing',
        'cool','nice','being','happy','shit','nothing','undo','night','rational',
        'neat','discipline','rocks','booring','kinda','real','rocking','uqtty','xxtx','yak',
        'sensible','kind','small','big','smart','object','goodboy','hello','queue',
        'faster','prateek','pruthviraj','pawar','mane','generate','font','model',
        'random','support','detection','zebra','new','whitner','umbrella','water']
testtext = ['rocks','nothingspecial','sssss','rigged','lostinsp','traingen',
            'patient','new','old','common','unique','highest','lowest']

allfiles = []

pagetype = ['ruled1','ruled2','blank']


def drawBoundingBox(imgcv,label,font,fontsize,linethickness,position,imgh,imgw,flpath):
    for labeltext in label:
        startposX = position[0]
        startposY = position[1]
        labelSize = cv2.getTextSize(labeltext,font,fontsize,linethickness)
        xmin = startposX           
        ymin = startposY
        xmax = xmin+labelSize[0][0] -1
        ymax = ymin+int(labelSize[0][1])    
        word_height = ymax - ymin                
        print("Word :",labeltext)
        ymax = ymax + math.ceil(word_height/3.0)
        cv2.putText(imgcv,labeltext,(xmin,ymax-math.ceil(word_height/3.0)),font,fontsize,(0,0,0),linethickness)
        #cv2.rectangle(imgcv,(xmin,ymin),(xmax,ymax),(255,0,0),1)
        for letter in list(labeltext):
            labelSize = cv2.getTextSize(letter,font,fontsize,linethickness)
            xmin = startposX
            ymin = startposY
            xmax = xmin+labelSize[0][0] - linethickness 
            ymax = ymin+int(labelSize[0][1])
            word_height = ymax - ymin
            word_width = xmax - xmin
            print("Letter",letter," Pos ",(xmin,ymin))
            print("letter prop: ",labelSize)
            #print(labelSize)
            ymax = ymax + math.ceil(word_height/3.0)
            #print("drawbox",xmin,ymin,xmax,ymax)   
            #cv2.rectangle(imgcv,(xmin-linethickness,ymin - 5),(xmax+linethickness,ymax),(0,0,255),1)
            datastring = flpath + ','+str(imgw)+','+str(imgh)+',' + 'text' + ','+ str(max((xmin-linethickness),0)) + ',' + str(max((ymin - 5),0)) + ',' + str(xmax+linethickness) + ',' + str(ymax) +'\n'
            datafile.write(datastring)
            startposX = xmax
            startposY = ymin
            #cv2.imshow('image',imgcv)                        
            #cv2.waitKey(0)         
            #cv2.destroyAllWindows() 
            
    return imgcv              

#drawBoundingBox(imgcv,["this is awesome"],font,2,[100,100])  

def generateData(folder = 'train'):
    global text,testtext,font, img_width, img_height,datafile, allfiles,testX,testY,trainX,trainY
    trainX = []
    trainY = []
    testX = []
    testY = []
    wordlist = []
    if folder == "train":
        # filepath = "C://Users//Nilesh//tensorflow//models//research//object_detection//images//train//"
        filepath = "images//train//"
        datafile = open('images//traindata.csv','w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+ 'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        datafile.write(datastring)
        num_of_images = num_of_train_images
        wordlist = text
    elif folder == "test":
        # filepath = "C://Users//Nilesh//tensorflow//models//research//object_detection//images//test//"
        filepath = "images//test//"
        datafile = open('images//testdata.csv','w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        datafile.write(datastring)
        num_of_images = num_of_test_images
        wordlist = testtext
        
    for count in range(num_of_images):
        
        num_of_words = np.random.randint(1,300)
        labellist = []
        startpos = [np.random.randint(20,40),np.random.randint(40,50)]
        xmin = startpos[0]
        ymin = startpos[1]
        positionList = []
        newfilepath = filepath + str(count)+'.jpg'
        fontchoice = np.random.choice(font)                                      
        fontsize = np.random.choice([1,1.5,2,2.5])
        linethickness= np.random.randint(1,3) 
        
        for i in range(num_of_words):
            word = np.random.choice(wordlist)
            wordSize = cv2.getTextSize(word,fontchoice,fontsize,linethickness)
            wordlength = wordSize[0][0]
            wordheight = int(wordSize[0][1])
            margin = np.random.randint(5,20)
            nymin = ymin - math.ceil(wordheight/3.0)
            xmax = xmin + wordlength
            ymax = ymin + wordheight 
            img_height = ymax + np.random.randint(20,50)
            img_width = xmax + np.random.randint(30,50)
            imgcv = np.zeros((img_height,img_width,3), np.uint8)
            fillcolor = np.random.randint(200,255)
            imgcv.fill(fillcolor)
            print("Word is : ",word)
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
        if len(indexList):
            if np.random.random() > 0.33 and len(indexList) > 50 :
                randomIndexList = np.random.choice(indexList,size = int(len(indexList)*0.5), replace = False)
            else:
                randomIndexList = indexList
                
            for index in randomIndexList:
                xmin = positionList[index][0]
                nymin = positionList[index][1]
                xmax = positionList[index][2]
                ymax = positionList[index][3]
                #print("generate",xmin,nymin,xmax,ymax)
                imgcv = drawBoundingBox(imgcv,[labellist[index]],fontchoice,fontsize,linethickness,[xmin,nymin],img_height,img_width,newfilepath)
                
                
            if np.random.random() > 0.5:
                imgcv = skimage.util.random_noise(imgcv, mode="gaussian")
                imgcv = np.array(fillcolor*imgcv,dtype='uint8')
                imgcv = cv2.blur(imgcv,(3,3))
                
            print(cv2.imwrite(filepath + str(count) + '.jpg', imgcv) )
            # print(labellist,positionList)
        
    if folder == 'train':
        return np.array(trainX),np.array(trainY)
    else:
        return np.array(testX), np.array(testY)
  

X,Y = generateData('train')

datafile.close()

                
            
            
            
        
        
    