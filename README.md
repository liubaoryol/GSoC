Human Activity Recognition Component of Robocomp

Code for recognition of human activities taken from several data sets
Start is done from CAD60 files, then generalization will be made to other open datasets.

First month:
1. Read and open dataset
2. Feature extraction. Following papers are the starting point:
https://cs.aston.ac.uk/~fariad/publications/ROMAN14_0187_FI.pdf
https://cs.aston.ac.uk/~fariad/publications/CC_DF_UN_NB@IROS16.pdf
https://cs.aston.ac.uk/~fariad/publications/ieee_roman15_dfaria.pdf
3. After implementing similar features set following the strategy mentioned in these papers, we will test a few classifiers. We can start with a multi-class linear SVM as mentioned in these papers.
4. improving the strategy (classifiers, LSTM, Random forests, ensembles, etc...). We can follow your ideas from your proposal

classes.py is the code where the class Person() is defined, which also has the necessary functions to read data activities from folders
read_data.py is where we create objects of class Person()
