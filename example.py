# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:35:16 2020

@author: Nilesh
"""

from Completedataset import *

# Create Instance of class to generate Dataset of Images for word detection
TrainDataset = ImageWordsDataset(10)

# Generate image objects which are to be used to generate dataset containing 
# letter boundingboxes
imgobjects = TrainDataset.generate_img_data(letterdata = True)

#Add noise to gaussian noise and blur to 50% of the images
TrainDataset.add_noise(0.5)

# Generate labels in csv format(as per tensorflow object detection api format)
TrainDataset.generatelabels("testdata.csv","images//","images//test//")

# Write images to folder- (location specified when generating labels)
TrainDataset.write_images_to_folder()

# LetterTrainDataset = letterBndBoxDataset(imgobjects)

# LetterTrainDataset.generate_labels(None, "images\\train\\")

# LetterTrainDataset.write_to_file()

# imgobjects[1].show_words(2)