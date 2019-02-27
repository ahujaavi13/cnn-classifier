# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:48 2019

@author: abhishek
"""

########################Import Reqisite Functions##########################

from run_specs import *
from helper_cnn import *
from model_builder import *
import pandas as pd

######################Create Data#########################################

test_data = create_test_data(TEST_DIR,IMG_SIZE)

######################Model Loading#########################################

model = tfModel(LR,IMG_SIZE,logging = wd +  "/Log/")
model.load(wd + "/Model/" + saved_model_name + ".tflearn")

#####################Generate Predictions##################################
save_location = wd + "/Predictions/" + predictions_file_name + ".csv"

prediction_list = [None]*len(test_data)
for i in range(0,len(test_data)):
	img = test_data[i]
	img_data, img_num = img
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	prediction = model.predict([data])[0]
	prediction_list[i] = prediction
	
file_names = os.listdir(TEST_DIR)
predictions = pd.DataFrame({"Files":file_names,"Probability without vs with helmet":prediction_list})
predictions.to_csv(save_location)





