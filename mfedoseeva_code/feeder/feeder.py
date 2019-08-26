# sys
import pickle

# torch
# import torch
# import torch.utils.data

import numpy as np
import os

class Feeder():
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V)
        label_path: the path to label
        center: If true, center at torso
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 #center=False,
                 mmap=False,
                 max_body = 1
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        #self.center = center
        self.mmap = mmap
        self.max_body = max_body

        self.load_data()
        
    def load_data(self):
        # data: N C V T 

        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
        else:
            raise ValueError()

        # load data
        if self.mmap == True:
            self.data = np.load(self.data_path,mmap_mode='r')
        else:
            self.data = np.load(self.data_path,mmap_mode=None) 

        # load num of valid frame length
        self.valid_frame_num = np.load(self.num_frame_path)

        # N - sample, C - xyz, T - frame, V - joint
        if self.max_body == 1 :
            self.N, self.C, self.T, self.V = self.data.shape
        else :
            raise NotImplementedError('multiperson not implemented')


    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        # input: C, T, V
        data_numpy = self.data[index]
       
        if self.mmap:
            data_numpy = np.array(data_numpy) # convert numpy.core.memmap.memmap to numpy


        label = self.label[index]
        # valid_frame_num = self.valid_frame_num[index]

        # if self.center == True:
        #     for t in range(valid_frame_num):
        #         # coords of the torso
        #         torso_coord = data_numpy[:, t, 2]
        #         for v in range(self.V):
        #             data_numpy[:, t, v] -= torso_coord

        return data_numpy, label



if __name__ == '__main__':

    import numpy as np

    # testing
    base_path = "../data0/CAD-60"
    environments = ['bathroom', 'bedroom', 'kitchen', 'livingroom', 'office']
    data_file = "train_data.npy"
    label_file = "train_label.pkl"
    num_frame_file = "train_num_frame.npy"
    
    for env in environments:

        print(f"Environment: {env}")

        data_path = os.path.join(base_path, env, data_file)
        label_path = os.path.join(base_path, env, label_file)
        num_frame_path = os.path.join(base_path, env, num_frame_file)

        dataset = Feeder(data_path, label_path, num_frame_path,
                         #center=False
                         )

        print('Labels distribution: ')
        oneh_vector = np.zeros(12)
        bins = np.bincount(dataset.label)
        for i in range(len(bins)):
            oneh_vector[i] = bins[i] 
        print(oneh_vector)

        # only print for small datasets
        for i, j in zip(dataset.sample_name, dataset.label):
            print(str(i) + ' | ' + str(j))

        print('Num of frames in samples: ')
        print(dataset.valid_frame_num)

        print('Samples statistics: ')
        print(f"{dataset.N} samples")
        print(f"{dataset.C} coords")
        print(f"{np.mean(dataset.valid_frame_num):.2f} average frames") 
        print(f"{dataset.V} joints")

        print ('------------------')












