# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:48 2019

@author: abhishek
"""

import subprocess

#The following code automatically detects root directory
wd  = subprocess.check_output('git rev-parse --show-toplevel', \
                              shell=True).decode('utf-8').strip()

#If you want to delete all previously stored logs and models
clear_log = "yes"
clear_model = "yes"

#######################Model Specifications######################

LR = 1e-3
IMG_SIZE = 64
MODEL_NAME = 'with_helmet-vs-without_helmet-convnet'

######################Create Data################################

datasource = "/home/ubuntu/Data_Warehouse/helmet-detection-data"
TRAIN_DIR = datasource + '/Train'
TEST_DIR = datasource + '/Test'
validation_perc = 10
epochs = 10

#################Model Saving####################################

saved_model_name = "helmet_detector"

#################Predictions generation##########################

predictions_file_name = "predictions"