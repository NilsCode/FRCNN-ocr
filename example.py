# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:35:16 2020

@author: Nilesh
"""

from Completedataset import *

# Create Instance of class to generate Dataset of Images for word detection - (Num Images = 10)
TrainDataset = ImageWordsDataset(200)
TestDataset =  ImageWordsDataset(50)
# Generate image objects which are to be used to generate dataset containing 
# letter boundingboxes
imgobjectstrain = TrainDataset.generate_img_data(letterdata = True,skip_percentage = 0.5,noise_add = 0.5,word_drawbndbox = False,letter_drawbndbox = False)
imgobjectstest = TestDataset.generate_img_data(letterdata = True,skip_percentage = 0.5,noise_add = 0.5,word_drawbndbox = False,letter_drawbndbox = False)

# Generate labels in csv format(as per tensorflow object detection api format)
TrainDataset.generatelabels("traindata.csv","images//","images//train//")
TestDataset.generatelabels("testdata.csv","images//","images//test//")

# Write images to folder- (location specified when generating labels)
TrainDataset.write_images_to_folder()
TestDataset.write_images_to_folder()

LetterTrainDataset = letterBndBoxDataset(imgobjectstrain)
LetterTestDataset = letterBndBoxDataset(imgobjectstest)

LetterTrainDataset.generate_labels('lettertraindata.csv',"letterimages//","letterimages//train//")
LetterTestDataset.generate_labels('lettertestdata.csv',"letterimages//","letterimages//test//")

LetterTrainDataset.write_to_file()
LetterTestDataset.write_to_file()

 