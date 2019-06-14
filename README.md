# Development of Human Activity Recognition Component of Robocomp

## Objectives

- [ ] Classify human activities with >95 % precision
- [ ] Extend classification to different datasets
- [ ] Extend classification to different environments
*Future development*
- [ ] Extend to several humans in one video
- [ ] Follow up of a person, i.e. give same unique id to a person

## Outline of tasks to follow 


First month (work on CAD-60):
1. Read and open dataset
2. Divide datasets to environments
2. *Feature extraction*. Starting point:

- https://cs.aston.ac.uk/~fariad/publications/ROMAN14_0187_FI.pdf
- https://cs.aston.ac.uk/~fariad/publications/CC_DF_UN_NB@IROS16.pdf
- https://cs.aston.ac.uk/~fariad/publications/ieee_roman15_dfaria.pdf
3. Choose the appropiated features using Multivariate Analysis
3. Test a few classifiers. Start with a multi-class linear SVM as mentioned in these papers.
4. Improve classifiers. Change DBMM base classifiers to my proposals.

Ideas for second month:
1. Join environments from CAD-60
2. Work now on other datasets.

Ideas for third month:
1. Improving code?
2. Tunning parameters, concluding project. 

## Datasets

| Dataset Name | Link | # subjects | # activities |  # joints |  # total samples | 
 | --- | --- | --- | --- | --- | --- |  
 | CAD-60 | [link](http://pr.cs.cornell.edu/humanactivities/data.php) | 4 | 12 | 15 | 60 | 
 | CAD-120 | --- | --- | --- | --- | --- |
 | UTKinect-Action | [link](http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html) | --- | --- | --- | --- |
 | Florence 3D-Action | [link](https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/) | 10 | 9 | --- | 215 | 
 | MSR Action3D | | [link](https://www.uow.edu.au/~wanqing/#Datasets) | 10 | 20 | 20 | 567 | 
 | MSR DailyActivity3D | [link](https://users.eecs.northwestern.edu/~jwa368/my_data.html) | --- | 20 | --- | 320 | 
 | SYSU | [link](http://isee.sysu.edu.cn/~hujianfang/ProjectJOULE.html) | 40 | 12 | 20 | 480 | 
 | UWA 3D Multiview II | [link](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html) | 10 | 30 | 20 | 1076 | 
 | SBU Kinect | [link](https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html) | 7 | 8 | 15 | 300 | 

## Articles


A Probalistic Approach for Human Everyday Activities Recognition using Body Motion from RGB-D Images, Diego R. Faria, Cristiano Premebida, Urbano Nunes. In IEEE RO-MAN'14: IEEE International Symposium on Robot and Human Interactive Communication, 2014.[link](https://research.aston.ac.uk/portal/en/researchoutput/a-probabilistic-approach-for-human-everyday-activities-recognition-using-body-motion-from-rgbd-images(73a87d76-bedb-480e-be74-9fab14d2dab8).html)

A Human Activity Recognition System Using Skeleton Data from RGBD Sensors, Enea Cippitelli, Samuele Gasparrini, Ennio Gambi, and Susanna Spinsante. In Computational Intelligence and Neuroscience, 2016. [link](https://www.researchgate.net/publication/298724252_A_Human_Activity_Recognition_System_Using_Skeleton_Data_from_RGBD_Sensors)

3D human action segmentation and recognition using pose kinetic energy, Junjie Shan, Srinivas Akella. IEEE International Workshop on Advanced Robotics and its Social Impacts, 2014.

Human Action Recognition by Representing 3D Skeletons as Points in a Lie Group, Raviteja Vemulapalli, Felipe Arrate and Rama Chellappa. CVPR 2014 [link](https://zpascal.net/cvpr2014/Vemulapalli_Human_Action_Recognition_2014_CVPR_paper.pdf)

A Human Activity Recognition System Based on Dynamic Clustering of Skeleton Data, Alessandro Manzi, Paolo Dario and Filippo Cavallo, 2017  [link](https://core.ac.uk/download/pdf/84495658.pdf)

## Explanation of scripts
classes.py -- code where class Person() is defined, with other necessary functions that are needed for class Person (read the data activities from folders)
read_data.py is where we create objects of class Person() and use its functions
To add: 
separation into environments. 
Feature extraction

## Questions
1. What is the extra file on the CAD-60 folders about
2. Details about on-fly-testing. Is it done on the same CAD-60 datasets? Has fly-testing used together with reinforcement learning?
3. How many layers does the ANN base classifier has in DBMM? (It has 40 neurons in hidden layers with hyperbolic tangent sigmoid activation function and normalized exponential (softmax)
4. What else would be done in second and third months?
