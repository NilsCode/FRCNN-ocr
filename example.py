# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:35:16 2020

@author: Nilesh
"""

from Completedataset import ImageWordsDataset

train = ImageWordsDataset(10)

train.generate_img_data()

train.imagesobject[0].show_image()
