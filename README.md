# Google Summer of Code 2019
## RoboComp
Development of the Human Activity Recognition component

## To use

 1. Download the CAD-60 dataset in the following [link](http://pr.cs.cornell.edu/humanactivities/data.php).
 2. Break dataset into five environments. Follow the readme file in the directory mfedoseeva. Code borrowed from [Mariyam Fedoseeva's github](https://github.com/mfedoseeva/GSOC19-har-project-robocomp) 
 You can train lstm or a cnn model.
 
 3.1 Train cnn algorithm using training_algorithms.train_cnn(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1), for example, training_algorithms.cnn("kitchen",50,1,30,".")
 3.2 TAnalogously you can train a lstm model using training_algorithms.train_lstm(...)
 4. You may apply transfer learning to some environments to improve accuracy. I recommend to use the base environment the kitchen since it has the best accuracy. To obtain a model just run one of the commands 3.1 or 3.2. To apply the code do training_algorithms.transfer_learning(har_model,X_train,y_train_num,X_test,y_test_num)
 5. To do ensemble learning do. ensemble_and_fit(n_members,environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1):
 
 
 

## Explanation of scripts
- classes.py -- code where class Person() is defined, with other necessary functions that are needed for class Person (read the data activities from folders)

- read_data.py is where we create objects of class Person() and use its functions

- functions.py in the folder feature_extraction contains help functions for selection and organization of features. feature_selection.py is the script that contains parafac and greedy unsupervised learning

- svm.py contains the script for plotting the confusion matrix

- training_algorithms.py is the main script, the one that you will be using. It contains all the training algorithm. Specifically, it contains the train_cnn, train_lstm, transfer learning and ensemble learning.
