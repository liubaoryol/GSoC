import argparse
import os
import random
import numpy as np
from feature_extraction.feature_extractor1 import extract_features_type1
from feature_extraction.tools import *
from feeder.feeder import Feeder
from feeder import utils
from support_operations.plot_confusion_matrix import plot_confusion_matrix

# for SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score

_ENVIRONMENT = ['office', 'livingroom', 'kitchen', 'bedroom', 'bathroom']
_SUBJECTS = 4
_CLASS_NAMES = ['talking on the phone', 'writing on whiteboard', 'drinking water', 'rinsing mouth with water', 'brushing teeth', 'wearing contact lenses', 'talking on couch', 'relaxing on couch', 'cooking (chopping)', 'cooking (stirring)', 'opening pill container', 'working on computer']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./data0/', help="directory of the dataset")

# HERE it is decided which dataset will be used
parser.add_argument('--dataset_name', default='CAD-60', help="dataset name ")
# this parameter controls if only cv accuracy is output or the model is fit and confusion matrices saved
parser.add_argument('--evaluation', default='cv', help='cv or full')


def fetch_data(params):
    # creates and returns feeder object with data
    if 'CAD-60' in params.dataset_name:
        params.data_feeder_args["data_path"] = params.dataset_dir+'/CAD-60'+ '/' + params.environment + '/train_data.npy'
        params.data_feeder_args["num_frame_path"] = params.dataset_dir+'/CAD-60' + '/' + params.environment +'/train_num_frame.npy'
        params.data_feeder_args["label_path"] = params.dataset_dir + '/CAD-60' + '/' + params.environment + '/train_label.pkl'
        dataset = Feeder(**params.data_feeder_args)
        return dataset
    else:
        raise NotImplementedError('only CAD-60 is supported')

def custom_cv_subj_fbf(valid_frame_num, subjects=_SUBJECTS):
    ''' 
    custom cross-validation rule, each fold = 1 subject
    '''
    n = len(valid_frame_num)
    actions = n/subjects
    
    frames_per_subject = np.zeros(subjects, dtype=int)
    for i in range(n):
        # subj from 0 to 3
        subj = int(i/actions)
        frames_per_subject[subj] += valid_frame_num[i]
    
    subj_end_idx = [None]*subjects

    prev = 0
    for i in range(subjects):
        subj_end_idx[i] = frames_per_subject[i] + prev
        prev += frames_per_subject[i] 

    start = 0
    for i in range(subjects):
        val_idx = np.arange(start, subj_end_idx[i])
        start += frames_per_subject[i]
        train_idx = [x for x in range(sum(valid_frame_num)) if x not in val_idx]
        yield train_idx, val_idx

def custom_cv_subj(data):
    ''' 
    custom cross-validation rule, each fold = 1 subject, not for frame by frame classification
    '''
    n, _ = data.shape
    subjects = _SUBJECTS
    # some actions are performed several times
    actions = 15
    for i in range(subjects):
        val_idx = np.arange(i * actions, i * actions + actions)
        train_idx = [x for x in range(subjects * actions) if x not in val_idx]
        yield train_idx, val_idx


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()

    params = utils.Params()

    params.dataset_dir = args.dataset_dir
    params.evaluation = args.evaluation
    # HERE dataset name is saved from the parser
    params.dataset_name = args.dataset_name

    # Set the random seed for reproducible experiments
    np.random.seed(0)
    random.seed(0)

    for env in _ENVIRONMENT:

        print('')

        params.environment = env
        print(f'Environment: {params.environment}')

        # get the data
        dataset = fetch_data(params)
        X = dataset.data
        Y = dataset.label
        num_frames = dataset.valid_frame_num

        X = frame_by_frame_samples(extract_features_type1(X, num_frames), num_frames)
        Y = frame_by_frame_labels(Y, num_frames)

        # print classes distribution
        print('Distribution of samples across label bins: ')
        oneh_vector = np.zeros(12)
        bins = np.bincount(Y)
        for i in range(len(bins)):
            oneh_vector[i] = bins[i] 
        print(oneh_vector)

        # define model
        clf = svm.SVC(decision_function_shape='ovo', gamma='scale')

        if params.evaluation == 'cv' or 'full':
            # runs cross validation and outputs accuracy
            clf = svm.SVC(decision_function_shape='ovo', gamma='scale')
            custom_cv = custom_cv_subj_fbf(num_frames)
            scores = cross_val_score(clf, X, Y, cv=custom_cv)
            print('Accuracy for each fold, i.e. subject')
            for i in range(4):
                print(scores[i])
            print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        if params.evaluation == 'full':
            # produces confusion matrices
            pred_labels = np.empty(len(Y), dtype=int)
            correct_labels = np.empty(len(Y), dtype=int)
            custom_cv = custom_cv_subj_fbf(num_frames)
            prev = 0
            for i in custom_cv:
                train_idx, test_idx = i
                X_train = X[train_idx]
                X_test = X[test_idx]
                Y_train = np.array(Y)[train_idx]
                Y_test = np.array(Y)[test_idx]
                
                Y_pred = clf.fit(X_train, Y_train).predict(X_test)
                pred_len = len(Y_pred)
                pred_labels[prev : (prev + pred_len)] = Y_pred
                correct_labels[prev : (prev + pred_len)] = Y_test
                prev = prev + pred_len
            np.set_printoptions(precision=2)
            plot_confusion_matrix(correct_labels, pred_labels, classes=np.array(_CLASS_NAMES), normalize=True, title=env)
        else:
            raise ValueError('wrong evluation parameter passed')


