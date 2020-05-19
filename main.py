# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:32:57 2020

@author: Nilesh
"""
	
# split into train and test set
from os import listdir
from numpy import asarray, zeros
from mrcnn.utils import Dataset
import pandas as pd
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.modelOri import MaskRCNN
import h5py
import numpy as np
import cv2

train = pd.read_csv('traindata.csv',header = None)
train.columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] 
test = pd.read_csv('testdata.csv',header = None)
test.columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

def read_many_hdf5(filename,num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images = []
    labelLink = {}
    
    # Open the HDF5 file
    file = h5py.File(filename, "r+")

    images = np.array(file["/images"]).astype("uint8")
    
    return images

class TextDataset(Dataset):
    
    def load_dataset(self, dataset_dir, is_train=True):
    # define one class
        self.add_class("dataset",1,"text")
        
        # define data locations
        images_dir = ""
        if is_train:
            images_dir = 'F:\\Projects\\Tensorflow\\FasterRcnn\\train\\'
        
        else:
            images_dir = 'F:\\Projects\\Tensorflow\\FasterRcnn\\test\\'
        
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
        
            img_path = images_dir + filename
        
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path)
        
    
    def extract_boxes(self,file):
        Row_list =[] 
        width = 416
        height = 416
        # Iterate over each row 
        a = file.rstrip().split('\\')
        if 'train' in a:
            for index, rows in train[train.filename == file].iterrows(): 
                # Create list for the current row 
                my_list =[rows.xmin, rows.ymin,rows.xmax, rows.ymax] 
                  
                # append the list to the final list 
                Row_list.append(my_list) 
              
            return Row_list,width,height
        else:
            for index, rows in test[test.filename == file].iterrows(): 
                # Create list for the current row 
                my_list =[rows.xmin, rows.ymin,rows.xmax, rows.ymax] 
                  
                # append the list to the final list 
                Row_list.append(my_list) 
              
            return Row_list,width,height
        
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        path = info['path']
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('text'))
        
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
    	info = self.image_info[image_id]
    	return info['path']
    
# train set
train_set = TextDataset()
train_set.load_dataset('text', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 
# test/val set
test_set = TextDataset()
test_set.load_dataset('text', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

	
# define a configuration for the model
class TextConfig(Config):
    
	# define the name of the configuration
	NAME = "text_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131
    
	
define image id
image_id = 0
mask, classids = train_set.load_mask(image_id)
load the image
image = test_set.load_image(image_id)
load the masks and the class ids
mask, class_ids = test_set.load_mask(image_id)
print(mask,class_ids)
extract bounding boxes from the masks
bbox = extract_bboxes(mask)
display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, test_set.class_names)


def startTraining():
# prepare config
    config = TextConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='F:\\Projects\\Tensorflow\\FasterRcnn\\', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
    
    

startTraining()

