from feature_extraction import tools
# import tools
import numpy as np

def extract_features_type1(X, num_frames):
	'''
	extracts 14 features
	centers the data at torso
	distance based features for (two hands and head), (two hands), (shoulders and feet), (hit and feet)
	temporal positional features for hands, elbows and head
	normalizes the data and rounds
	out_shape (N, 14. n_frames)
	''' 
	# N - samples, C - xyz, T - frames(all 2K for now, use num_frames to get actual length of a sample), V - joints 
	N, C, T, V = X.shape
	# center all data first
	X = tools.center(X, num_frames)
	# we will use 14 features this time
	features = np.zeros((N, 14, T))
	# dist between two hands 11, 12 and a head 0
	features[:, 0, :] = tools.dist_to_joint_allsamples(
    X[:, :, :, 11], X[:, :, :, 0], num_frames) + tools.dist_to_joint_allsamples(
    X[:, :, :, 12], X[:, :, :, 0], num_frames)
    # dist between two hands 11, 12
	features[:, 1, :] = tools.dist_to_joint_allsamples(X[:, :, :, 11], X[:, :, :, 12], num_frames)
	# dist between shoulders and feet 3, 5, 13, 14
	features[:, 2, :] = tools.dist_to_joint_allsamples(
    X[:, :, :, 3], X[:, :, :, 13], num_frames) + tools.dist_to_joint_allsamples(
    X[:, :, :, 5], X[:, :, :, 14], num_frames)
    # dist between hip and feet 7, 9, 13, 14
	features[:, 3, :] = tools.dist_to_joint_allsamples(X[:, :, :, 7], X[:, :, :, 13], num_frames) + tools.dist_to_joint_allsamples(X[:, :, :, 9], X[:, :, :, 14], num_frames)
    # temporal positional features
	# left hand 11 x, y
	features[:, 4, :] = tools.diff_position_x(X[:, :, :, 11], num_frames)
	features[:, 5, :] = tools.diff_position_y(X[:, :, :, 11], num_frames)
	# right hand 12
	features[:, 6, :] = tools.diff_position_x(X[:, :, :, 12], num_frames)
	features[:, 7, :] = tools.diff_position_y(X[:, :, :, 12], num_frames)
	# left elbow 4
	features[:, 8, :] = tools.diff_position_x(X[:, :, :, 4], num_frames)
	features[:, 9, :] = tools.diff_position_y(X[:, :, :, 4], num_frames)
	# right elbow 6
	features[:, 10, :] = tools.diff_position_x(X[:, :, :, 6], num_frames)
	features[:, 11, :] = tools.diff_position_y(X[:, :, :, 6], num_frames)
	# right head 0
	features[:, 12, :] = tools.diff_position_x(X[:, :, :, 0], num_frames)
	features[:, 13, :] = tools.diff_position_y(X[:, :, :, 0], num_frames)
	# normalize and round
	features = tools.normalize(features, num_frames)
	features = np.around(features, decimals=3)

	return features


if __name__ == '__main__':

	import sys
	sys.path.insert(0, '../feeder')
	from feeder import Feeder

	# a simple test

	data_path = "../data0/CAD-60/office/train_data.npy"
	label_path = "../data0/CAD-60/office/train_label.pkl"
	num_frame_path = "../data0/CAD-60/office/train_num_frame.npy"
	dataset = Feeder(data_path, label_path, num_frame_path)

	X = dataset.data
	num_frame = dataset.valid_frame_num

	features = extract_features_type1(X, num_frame)

	print(features[15, :, :10])