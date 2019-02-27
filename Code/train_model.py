# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:48 2019

@author: abhishek
"""

########################Import Reqisite Packages##########################

from run_specs import * 
from helper_cnn import *
from model_builder import *
import os

#######################Remove old Logs and Models#######################

if clear_log == "yes":
	folder = wd +  "/Log/" + MODEL_NAME
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e + " does not exist")
			
if clear_model == "yes":
	folder = wd +  "/Model/" 
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e + " does not exist")



######################Create Data#########################################

train_data = create_train_data(TRAIN_DIR,IMG_SIZE)
x_train,y_train,x_test,y_test = np_data(train_data,IMG_SIZE, validation_perc = validation_perc)

######################Model Fitting#########################################

model = tfModel(LR,IMG_SIZE,logging = wd +  "/Log/")
model.fit({'input': x_train}, \
	  {'targets': y_train}, n_epoch=epochs, \
          validation_set=({'input': x_test}, \
			  {'targets': y_test}), \
    	  snapshot_step=500, \
	  show_metric=True, \
	  run_id=MODEL_NAME)

model.save(wd + "/Model/" + saved_model_name + ".tflearn")

print("Model Training Complete")
