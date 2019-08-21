import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
sys.path.append("feature_extraction")
sys.path.append("data_preparation")
sys.path.append("classification")
from keras.utils import to_categorical#one-hot encode target column
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.callbacks import ModelCheckpoint
import functions
import classes
import preprocessing_functions
import svm
import preprocessing_functions
import matplotlib.pyplot as plt

def train_cnn(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1):
	if not os.path.exists(out_dir):
		os.system('mkdir '+ out_dir)
		
	if type(n_units)==int:
		n_units =[n_units]
	#len ( n_units ) = n_layers.
	person1,person2,person3,person4 = preprocessing_functions.create_person_instances(environment)
	
	#centering torso and normalizing by height. 
	preprocessing_functions.standardize_person(person1),  preprocessing_functions.standardize_person(person2),  preprocessing_functions.standardize_person(person3),  preprocessing_functions.standardize_person(person4)

	#augmenting
	"""person1.pos_activities+=functions.activities_data_augmentation_rotation(person1.pos_activities)
	person2.pos_activities+=functions.activities_data_augmentation_rotation(person2.pos_activities)
	person3.pos_activities+=functions.activities_data_augmentation_rotation(person3.pos_activities)
	person4.pos_activities+=functions.activities_data_augmentation_rotation(person4.pos_activities)
	person1.label+=person1.label
	person2.label+=person2.label
	person3.label+=person3.label
	person4.label+=person4.label
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
		preprocessing_functions.clean(person1_features,person2_features,person3_features,person4_features)
		
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
		model.add(MaxPooling2D(pool_size=2))
		model.add(Dropout(0.3))
	model.add(Flatten())
	model.add(Dense(len(set(y_test)), activation='softmax'))
	#compile model using accuracy to measure model performance
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	#save the best model
	filepath= out_dir + "/" + str(n_layers)+"-layer_"+str(n_units)+"_unit_"+environment+"cnn-BestModel.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#train the model
	history = model.fit(X_train, y_train_num, validation_data=(X_test, y_test_num), epochs=50, callbacks = callbacks_list)
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
	


def train_lstm(environment,subvideo_frames,n_layers,n_units,n_subvideo_features=-1):
	#len(lstm_units) = n_layers. List of units per layer
	person1,person2,person3,person4 = preprocessing_functions.create_person_instances(environment)
	
	#centering torso and normalizing by height. 
	preprocessing_functions.standardize_person(person1),  preprocessing_functions.standardize_person(person2),  preprocessing_functions.standardize_person(person3),  preprocessing_functions.standardize_person(person4)

	#-1 stands that the features for each frame is in the feature vector of the activity
	person1_features = functions.activities_feature_vector(person1.pos_activities,n_subvideo_features)
	person2_features = functions.activities_feature_vector(person2.pos_activities,n_subvideo_features)
	person3_features = functions.activities_feature_vector(person3.pos_activities,n_subvideo_features)
	person4_features = functions.activities_feature_vector(person4.pos_activities,n_subvideo_features)

	if environment == "all":
		env = ""
	else:
		env = environment + "/"


	#contruct training set and test set

	if not env:
		preprocessing_functions.clean(person1_features,person2_features,person3_features,person4_features)


	parted_act1 = functions.partition_activities(person1_features, subvideo_frames)
	parted_act2 = functions.partition_activities(person2_features, subvideo_frames)
	parted_act3 = functions.partition_activities(person3_features, subvideo_frames)
	parted_act4 = functions.partition_activities(person4_features, subvideo_frames)

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




	labelencoder = LabelEncoder()
	labelencoder.fit(person1.label)

	y_train_num = labelencoder.fit_transform(y_train)
	y_test_num = labelencoder.fit_transform(y_test)

	y_train_num = to_categorical(y_train_num)
	y_test_num = to_categorical(y_test_num)


	# Train neural network. SVM, random forest, etc. Artificial Neural Network. LSTM


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






