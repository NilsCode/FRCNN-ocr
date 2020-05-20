# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:35:16 2020

@author: Nilesh
"""

from Completedataset import *

# Create Instance of class to generate Dataset of Images for word detection - (Num Images = 10)
TrainDataset = ImageWordsDataset(800)
TestDataset =  ImageWordsDataset(100)
# Generate image objects which are to be used to generate dataset containing 
# letter boundingboxes
imgobjectstrain = TrainDataset.generate_img_data(letterdata = False,skip_percentage = 0.5,noise_add = 0.5,word_drawbndbox = True,letter_drawbndbox = False)
imgobjectstest = TestDataset.generate_img_data(letterdata = False,skip_percentage = 0.5,noise_add = 0.5,word_drawbndbox = True,letter_drawbndbox = False)

# Generate labels in csv format(as per tensorflow object detection api format)
TrainDataset.generatelabels("traindata.csv","images//","images//train//")
TestDataset.generatelabels("testdata.csv","images//","images//test//")

# Write images to folder- (location specified when generating labels)
TrainDataset.write_images_to_folder()
TestDataset.write_images_to_folder()
# LetterTrainDataset = letterBndBoxDataset(imgobjects)

# LetterTrainDataset.generate_labels(None, "images\\train\\")

# LetterTrainDataset.write_to_file()

# imgobjects[1].show_words(2)