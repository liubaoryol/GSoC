import sys
import os
import numpy as np
sys.path.append("feature_extraction")
sys.path.append("data_preparation")

import functions
import classes

#Obtain and save joint points and activity of person
# set working directory
dir1 = "/home/lytica/Documents/GSoC/GSoC/separated_data/bedroom/data1"
dir2 = "/home/lytica/Documents/GSoC/GSoC/separated_data/bedroom/data2"
dir3 = "/home/lytica/Documents/GSoC/GSoC/separated_data/bedroom/data3"
dir4 = "/home/lytica/Documents/GSoC/GSoC/separated_data/bedroom/data4"


person1=classes.Person()
person1.read_activity_from_folder(dir1)

person2=classes.Person()
person2.read_activity_from_folder(dir2)

person3=classes.Person()
person3.read_activity_from_folder(dir3)

person4=classes.Person()
person4.read_activity_from_folder(dir4)

#person1.activity
#person1.label

person1.pos_activities = functions.normalize_by_height_all(functions.center_torso(functions.pos(person1.activity)))
person2.pos_activities = functions.normalize_by_height_all(functions.center_torso(functions.pos(person2.activity)))
person3.pos_activities = functions.normalize_by_height_all(functions.center_torso(functions.pos(person3.activity)))
person4.pos_activities = functions.normalize_by_height_all(functions.center_torso(functions.pos(person4.activity)))

'''
There are several ways for viewing an activity.
An activity variable is an array of 12 activities for each person.
Each activity is composed of n number of frames. Each frame has 170 entries: 
Frame#,ORI(1),P(1),ORI(2),P(2),...,P(11),J(11),P(12),...,P(15)
where ORI(i)-- orientation of joint has 10 entries
P(i) position of joint has 4 entries.
We have 15 joints, 11 of them given together with orientation

Therefore it occurs to me three ways to represent the data:
1. Flat list (no confidence value)
2. List of lists composed ONLY with positions of joints (no confidence value)
3. List of lists composed with both position and orientation of joints in same/different sublists (no confidence value)
4. Same as above but together with confidence value


'''                                                       


#feature extractor. 
#Get the features for each activity.
#Add features that you are interested in, the ones that would describe the WHOLE activity. Adapt coordinate system. Instead of handcrafting the features, use PCA

person1_features = functions.activities_feature_vector(person1.pos_activities)
person2_features = functions.activities_feature_vector(person2.pos_activities)
person3_features = functions.activities_feature_vector(person3.pos_activities)
person4_features = functions.activities_feature_vector(person4.pos_activities)

#contruct training set and test set
#Removing first the "none" element
person1_features.pop(0)
person2_features.pop(-2)
person3_features.pop(-2)
person4_features.pop(-2)

person1.label.pop(0)
person2.label.pop(-2)
person3.label.pop(-2)
person4.label.pop(-2)

#removing the random activity
person1_features.pop(10)
person2_features.pop(10)
person3_features.pop(2)
person4_features.pop(-5)

person1.label.pop(10)
person2.label.pop(10)
person3.label.pop(2)
person4.label.pop(-5)


X_train = np.array(person1_features + person2_features + person3_features)
X_test = np.array(person4_features)
y_train = person1.label + person2.label + person3.label
y_test =  person4.label

#reshape data to fit model
X_train = X_train.reshape(len(X_train),X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(len(X_test),X_train.shape[1],X_train.shape[2],1)

from keras.utils import to_categorical#one-hot encode target column
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
y_train_num = labelencoder.fit_transform(y_train)
y_test_num = labelencoder.fit_transform(y_test)



y_train_num = to_categorical(y_train_num)
y_test_num = to_categorical(y_test_num)

y_train_num[0]


# Train neural network. SVM, random forest, etc. Artificial Neural Network. LSTM
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#create model

model = Sequential()#add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)))
#model.add(Conv2D(8, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(len(set(y_test)), activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train_num, validation_data=(X_test, y_test_num), epochs=500)




