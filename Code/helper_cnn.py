# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:48 2019

@author: abhishek
"""

import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm 

def create_label(img):
    word_label = img.split('.')[0]
    if word_label == 'without_helmet':
        return np.array([1,0])
    elif word_label == 'with_helmet':
        return np.array([0,1])
    
def create_train_data(TRAIN_DIR,IMG_SIZE):
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    return training_data

def create_test_data(TEST_DIR,IMG_SIZE):
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num=img.split('.')[0]
        path=os.path.join(TEST_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    return testing_data

def np_data(train_data,IMG_SIZE,validation_perc):
    
    validation_count = int(round(((validation_perc/100)*len(train_data)),0))
    
    train = train_data[:-validation_count]
    test = train_data[-validation_count:]
    
    x_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = [i[1] for i in train]
    
    x_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = [i[1] for i in test]
    return x_train,y_train,x_test,y_test
