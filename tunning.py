import cnn_algorithm 
import lstm_algorithm

	
# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy


#All environments for which to tune
environments = ["kitchen","office","bedroom","bathroom","livingroom"]

#space of hyperparameters to evaluate 
subvideo_frames = [10,25,50,75]
n_layers = [1,2]
cnn_units_1 = [[8],[16],[32]]
cnn_units_2 = [[16,8],[32,16]]
lstm_units = [[10],[20],[50],[100],[150]]


#tunning parameters for cnn
for environment in environments:
	for i in subvideo_frames:
		for j in n_layers:
			if j == 1:
				for k in cnn_units_1:
					cnn_algorithm.train_cnn(environment,i,j,k)
			if j == 2:
				for k in cnn_units_2:
					cnn_algorithm.train_cnn(environment,i,j,k)

#tunning parameters for lstm
for environment in environments:
	for i in subvideo_frames:
		for j in lstm_units:
			lstm_algorithm.train_lstm(environment,i,1,j)
