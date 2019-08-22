import sys
import os
import numpy as np
sys.path.append("feature_extraction")
sys.path.append("data_preparation")
sys.path.append("classification")
import functions
import classes
import preprocessing_functions
import svm
import preprocessing_functions
import matplotlib.pyplot as plt
from numpy import argmax
from keras.utils import to_categorical#one-hot encode target column
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM,MaxPooling2D,Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate


def get_train_test(environment,subvideo_frames,subvideo_features = -1):
	
	person1,person2,person3,person4 = preprocessing_functions.create_person_instances(environment)
	
	#centering torso and normalizing by height. 
	preprocessing_functions.standardize_person(person1),  preprocessing_functions.standardize_person(person2),  preprocessing_functions.standardize_person(person3),  preprocessing_functions.standardize_person(person4)

	#augmenting data (rotating to #radians)
	"""person1.pos_activities+=functions.activities_data_augmentation_rotation(person1.pos_activities)
	person2.pos_activities+=functions.activities_data_augmentation_rotation(person2.pos_activities)
	person3.pos_activities+=functions.activities_data_augmentation_rotation(person3.pos_activities)
	person4.pos_activities+=functions.activities_data_augmentation_rotation(person4.pos_activities)
	person1.label+=person1.label
	person2.label+=person2.label
	person3.label+=person3.label
	person4.label+=person4.label"""
	"""
	person1.pos_activities+=functions.activities_data_augmentation_rotation(person1.pos_activities,2)
	person2.pos_activities+=functions.activities_data_augmentation_rotation(person2.pos_activities,2)
	person3.pos_activities+=functions.activities_data_augmentation_rotation(person3.pos_activities,2)
	person4.pos_activities+=functions.activities_data_augmentation_rotation(person4.pos_activities,2)
	person1.label+=person1.label
	person2.label+=person2.label
	person3.label+=person3.label
	person4.label+=person4.label"""



	person1_features = functions.activities_feature_vector(person1.pos_activities,subvideo_features)
	person2_features = functions.activities_feature_vector(person2.pos_activities,subvideo_features)
	person3_features = functions.activities_feature_vector(person3.pos_activities,subvideo_features)
	person4_features = functions.activities_feature_vector(person4.pos_activities,subvideo_features)

	if environment == "all":
		env = ""
	else:
		env = environment 


	#contruct training set and test set

	if not env:
		preprocessing_functions.clean_feat(person1_features,person2_features,person3_features,person4_features)
		preprocessing_functions.clean_lab(person1,person2,person3,person4)
		
	parted_act1 = functions.partition_activities(person1_features,subvideo_frames)
	parted_act2 = functions.partition_activities(person2_features,subvideo_frames)
	parted_act3 = functions.partition_activities(person3_features,subvideo_frames)
	parted_act4 = functions.partition_activities(person4_features,subvideo_frames)

	new_labels1 = functions.multiplicate_labels(parted_act1,person1.label)
	new_labels2 = functions.multiplicate_labels(parted_act2,person2.label)
	new_labels3 = functions.multiplicate_labels(parted_act3,person3.label)
	new_labels4 = functions.multiplicate_labels(parted_act4,person4.label)

	X_train = np.concatenate(np.array(parted_act1 + parted_act2 +parted_act3))
	X_test = np.concatenate(np.array(parted_act4))
	y_train = new_labels1 + new_labels2 + new_labels3
	y_test = new_labels4

	#sc = StandardScaler()
	#sc.fit(person1_features+person2_features+person3_featu)
	#X_train = sc.transform(X_train)
	#X_test = sc.transform(X_test)


	#reshape data to fit model
	X_train = X_train.reshape(len(X_train),X_train.shape[1],X_train.shape[2],1)
	X_test = X_test.reshape(len(X_test),X_train.shape[1],X_train.shape[2],1)

	
	return X_train,y_train,X_test,y_test,person1

def train_cnn(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1):
	
	if not os.path.exists(out_dir):
		os.system('mkdir '+ out_dir)
		
	if type(n_units)==int:
		n_units =[n_units]
	#len ( n_units ) = n_layers.

	X_train,y_train,X_test,y_test,person1 = get_train_test(environment,subvideo_frames,subvideo_features)
	labelencoder = LabelEncoder()
	labelencoder.fit(person1.label)
	y_train_num = labelencoder.fit_transform(y_train)
	y_test_num = labelencoder.fit_transform(y_test)

	y_train_num = to_categorical(y_train_num)
	y_test_num = to_categorical(y_test_num)
	#create model
	model = Sequential()
	model.add(Conv2D(n_units[0], kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)))
	for i in range(n_layers-1):
		model.add(Conv2D(n_units[i+1], kernel_size=3, activation='relu'))
		#model.add(MaxPooling2D(pool_size=2))
		#model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(len(set(y_test)), activation='softmax'))
	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	#save the best model
	filepath= out_dir + "/" + str(n_layers)+"-layer_"+str(n_units)+"_unit_"+environment+"cnn-BestModel.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#train the model
	history = model.fit(X_train, y_train_num, validation_data=(X_test, y_test_num), epochs=10, callbacks = callbacks_list)
	preprocessing_functions.save_history_training(history,"cnn",n_layers,n_units,out_dir)
	#load the best saved model
	model = Sequential()
	model.add(Conv2D(n_units[0], kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)))
	for i in range(n_layers-1):
		model.add(Conv2D(n_units[i+1], kernel_size=3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(len(set(y_test)), activation='softmax'))
	model.load_weights(filepath)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# save the confusion matrix
	Y_pred = model.predict(X_test)
	y_pred_labels = np.argmax(Y_pred, axis=1)
	y_test_num2 = np.argmax(y_test_num,axis=1)

	scores = model.evaluate(X_test, y_test_num, verbose=0)
	
	acc ="%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

	confusion_name = out_dir +"/"+ str(n_layers)+"-layer_"+str(n_units)+"unit"+environment+"cnn-ConfusionMatrix" + acc 
	svm.plot_confusion_matrix(y_test_num2, y_pred_labels, person1.label, confusion_name,
		                  normalize=True,
		                  title=environment)
	plt.close()
	return model, X_train, y_train_num, X_test, y_test_num


def train_lstm(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features=-1):
	
	if not os.path.exists(out_dir):
		os.system('mkdir '+ out_dir)
		
	if type(n_units)==int:
		n_units =[n_units]
	#len ( n_units ) = n_layers.

	X_train,y_train,X_test,y_test,person1 = get_train_test(environment,subvideo_frames,subvideo_features)
	X_train = X_train.reshape([X_train.shape[0],X_train.shape[1],X_train.shape[2]])
	X_test = X_test.reshape([X_test.shape[0],X_test.shape[1],X_test.shape[2]])
	labelencoder = LabelEncoder()
	labelencoder.fit(person1.label)
	y_train_num = labelencoder.fit_transform(y_train)
	y_test_num = labelencoder.fit_transform(y_test)

	y_train_num = to_categorical(y_train_num)
	y_test_num = to_categorical(y_test_num)



	labelencoder = LabelEncoder()
	labelencoder.fit(person1.label)

	y_train_num = labelencoder.fit_transform(y_train)
	y_test_num = labelencoder.fit_transform(y_test)

	y_train_num = to_categorical(y_train_num)
	y_test_num = to_categorical(y_test_num)
	#create model

	model = Sequential()
	model.add(LSTM(n_units[0],input_shape = (X_train.shape[1],X_train.shape[2])))
	for i in range(n_layers-1):
		model.add(LSTM(n_units[i+1]))
	model.add(Dropout(0.2))
	model.add(Dense(len(set(y_test)), activation='softmax'))


	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	#save the best model
	filepath=out_dir + "/" + str(n_layers)+"-layer_"+str(n_units)+"unit"+environment+"lstm-BestModel.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#train the model
	history = model.fit(X_train, y_train_num, validation_data=(X_test, y_test_num), epochs=50, callbacks = callbacks_list)
	#plot history and save image
	preprocessing_functions.save_history_training(history,"lstm",n_layers,n_units, out_dir)

	model = Sequential()
	model.add(LSTM(n_units[0],input_shape = (X_train.shape[1],X_train.shape[2])))
	for i in range(n_layers-1):
		model.add(LSTM(n_units[i+1]))
	model.add(Dropout(0.2))
	model.add(Dense(len(set(y_test)), activation='softmax'))
	model.load_weights(filepath)
	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	#confusion matrix
	Y_pred = model.predict(X_test)
	y_pred_labels = np.argmax(Y_pred, axis=1)
	y_test_num1 = np.argmax(y_test_num,axis=1)
	
	scores = model.evaluate(X_test, y_test_num, verbose=0)
	acc ="%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
	confusion_name = out_dir + "/" + str(n_layers)+"-layer_"+str(n_units)+"unit"+environment+"lstm-ConfusionMatrix" + acc
	svm.plot_confusion_matrix(y_test_num1, y_pred_labels, person1.label, confusion_name,
		                  normalize=True,
		                  title=environment)

	plt.close()
	return model


#Gives amazing results!!
def transfer_learning(har_model,X_train,y_train_num,X_test,y_test_num):
	#Transfer learning where we define weights to star the learning from. Weights are not freezed
	model = Sequential()
	model.add(har_model)
	#model.add(Flatten())
	#model.add(Dense(13, activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, y_train_num, validation_data=(X_test, y_test_num), epochs=50)


#The following code is for ensemble learning
# load models from file
def get_models(n_models,environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		model = train_cnn(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features)
		# add to list of members
		all_models.append(model)
	return all_models
 
# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(members[0].output_shape[1], activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	#plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# fit a stacked model
def fit_stacked_model(model, inputX, inputy,testX,testy,epochs=100):
	# the output should be already encoded
	# prepare input data
	X_train = [inputX for _ in range(len(model.input))]
	X_test = [testX for _ in range(len(model.input))]
	
	# fit model
	model.fit(X_train, inputy, epochs, validation_data=(X_test, testy),verbose=0)
 
# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)

#This functions is to create a stacked model of cnn. Ensemblence of models improve quality
def ensemble_and_fit(n_members,environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1):
	all_models = get_models(n_members,environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features)
	ens_model = define_stacked_model(all_models)
	fit_stacked_model (ens_model,X_train,y_train_num,X_test,y_test_num)
	return ens_model

