# Google Summer of Code 2019 for RoboComp
Development of the Human Activity Recognition component

## To use

 1. Clone the repository. The needed data separated into environments is located in the folder separated_data. The whole dataset used for this work can be found in the following [link](http://pr.cs.cornell.edu/humanactivities/data.php)
 
 You can train LSTM or a CNN model.
 
 3. Train cnn algorithm using 
 
```commandline
training_algorithms.train_cnn(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1)
```
For example:


```commandline
training_algorithms.cnn("kitchen",50,1,30,".")
```
The options for environment are: "kitchen", "livingroom", "bathroom", "bedroom", "office" or "all".
 
 Analogously you can train a lstm model using 
 
```commandline
training_algorithms.train_lstm(environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1)
```
 Both commands output model, X_train, y_train, X_test, y_test
 
 4. You may apply transfer learning to some environments to improve accuracy. I recommend to use the base environment the kitchen since it has the best accuracy. To obtain a model just run one of the commands 3.1 or 3.2. To apply the code do 
 
```commandline
training_algorithms.transfer_learning(har_model,X_train,y_train_num,X_test,y_test_num)
```
here the variable har_model is obtained from training a previous model with train_cnn or train_lstm. These functions output the model, which then will be reused for transfer learning
 
 5. To do ensemble learning run
 
```commandline
ensemble_and_fit(n_members,environment,subvideo_frames,n_layers,n_units,out_dir,subvideo_features = -1)
```
n_members is the number of neural networks you want to ensemble
 
 

## Explanation of scripts
- classes.py -- code where class Person() is defined, with other necessary functions that are needed for this class ( for example read the data activities from folders)

- read_data.py is where we create objects of class Person() and use its functions

- functions.py in the folder feature_extraction contains help functions for selection and organization of features. feature_selection.py is the script that contains parafac and greedy unsupervised learning

- svm.py contains the script for plotting the confusion matrix

- training_algorithms.py is the main script, the one that you will be using. It contains all the training algorithms. Specifically, it contains the train_cnn, train_lstm, transfer learning and ensemble learning.
