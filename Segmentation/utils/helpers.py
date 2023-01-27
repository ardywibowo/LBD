import cv2
import numpy as np
import itertools
import operator
import os, csv
import tensorflow as tf

import time, datetime

def pavpu(accurate_mask, uncertainty_map, uncertainty_threshold, acc_threshold=0.5):
    
    height = accurate_mask.shape[0]
    width = accurate_mask.shape[1]
    
    v_patches = np.floor(height/4)
    h_patches = np.floor(width/4)
    
    uncertainty = np.zeros([v_patches, h_patches])
    accurate = np.zeros([v_patches, h_patches])
    
    p_ac = 0
    p_ui = 0
    pavpu = 0
    n=v_patches*h_patches
    for i in range(v_patches):
        for j in range(h_patches):
            uncertainty_patch = uncertainty_map[i*4 : (i + 1)*4][j*4 : (j + 1)*4]
            accurate_patch = accurate_mask[i*4 : (i + 1)*4][j*4 : (j + 1)*4]
            

            uncertainty[i][j] = np.mean(uncertainty_patch) > uncertainty_threshold
            accurate[i][j] = np.mean(accurate_patch) > acc_threshold
            
    n_ac = np.sum(np.logical_not(uncertainty) * accurate)
    n_c =  np.sum(np.logical_not(uncertainty))
            
    n_iu = np.sum(np.logical_not(accurate) * uncertainty)
    n_i =  np.sum(np.logical_not(accurate))
            
    if n_c>0:
        p_ac += float(n_ac) / float(n_c)
    if n_i>0:
        p_ui += float(n_iu) / float(n_i)
    if n_c>0 or n_i>0:
        pavpu += float(n_ac + n_iu) / float(n)
    
    return p_ac, p_ui, pavpu

def pavpu_labels(prediction_mask, label_mask, ignore_label,uncertainty_map, uncertainty_threshold, acc_threshold=0.5, filter_size=4):
    
    accurate_mask = np.equal(prediction_mask,label_mask).astype('float')
    
    not_ignore_mask = 1.0 - np.equal(label_mask,ignore_label).astype('float')
    
    height = accurate_mask.shape[0]
    width = accurate_mask.shape[1]
    
    v_patches = int(np.floor(height/filter_size))
    h_patches = int(np.floor(width/filter_size))
    
    uncertainty = np.zeros([v_patches, h_patches])
    accurate = np.zeros([v_patches, h_patches])
    
    p_ac = 0.
    p_ui = 0.
    pavpu = 0.
    
    skipped=0
    n=0
    
    for i in range(v_patches):
        for j in range(h_patches):
            
          not_ignore_patch = not_ignore_mask[i*filter_size : (i + 1)*filter_size][j*filter_size : (j + 1)*filter_size]
            
          if np.sum(not_ignore_patch)>0:
            
            uncertainty_patch = uncertainty_map[i*filter_size : (i + 1)*filter_size][j*filter_size : (j + 1)*filter_size]
            accurate_patch = accurate_mask[i*filter_size : (i + 1)*filter_size][j*filter_size : (j + 1)*filter_size]
            
            
            

            uncertainty[i][j] = np.sum(uncertainty_patch*not_ignore_patch)/np.sum(not_ignore_patch) > uncertainty_threshold
            accurate[i][j] = np.sum(accurate_patch*not_ignore_patch)/np.sum(not_ignore_patch) > acc_threshold
            n+=1
          else:
              skipped += 1
              continue
    n_ac = np.sum(np.logical_not(uncertainty) * accurate)
    n_c =  np.sum(np.logical_not(uncertainty)) - skipped
            
    n_iu = np.sum(np.logical_not(accurate) * uncertainty)
    n_i =  np.sum(np.logical_not(accurate)) - skipped
            
    if n_c>0:
        p_ac += float(n_ac) / float(n_c)
    if n_i>0:
        p_ui += float(n_iu) / float(n_i)
    if n_c>0 or n_i>0:
        pavpu += float(n_ac + n_iu) / float(n)

    return p_ac, p_ui, pavpu



def stable_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 3D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """

    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def one_hot_it_gt(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """

    semantic_map = []
    for colour in label_values:
        class_map = np.equal(label, colour)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """

    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """

    
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

