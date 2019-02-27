# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:48 2019

@author: abhishek
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#Here, building a CNN with multiple layers. It can be customized according to model performance
def tfModel(LR,IMG_SIZE,logging):
    tf.reset_default_graph()
    
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')       #32 is the number of filters and 5 is 5x5 size of filter
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = fully_connected(convnet, 1024, activation='relu')      
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir=logging, tensorboard_verbose=3)
    return model