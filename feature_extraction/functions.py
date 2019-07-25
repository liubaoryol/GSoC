#import numba
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
#import umap
import numpy as np
import math

#@numba.njit()
def metric(act1,act2):
	#only one activity. This metric is not good at all
	dist=0
	min_dim = min(len(act1),len(act2))
	for i in range(min_dim):
		for j in range(len(act1[i])):
			dist +=distance.euclidean(act1[i][j],act2[i][j])
	return dist

' As fist approach we will use only positions of joints, with each joint grouped in a 3D array.'
def pos(activities): 
	#activities are all the activities that a person has made
	pos_act = []
	for array in activities:
		pos_act.append(np.array(array))
	cols = [] #Here we will save the positional cols
	#Finding positions for the first 11 joints
	for i in range(0,len(pos_act[0][0])-16,14):
		cols.append([i+11,i+12,i+13]) 
	cols=cols[:-1]
	#Finding positions for the last 4 joints
	for i in range(len(pos_act[0][0])-16,len(pos_act[0][0]),4):
		cols.append([i,i+1,i+2])
	for j in range(len(pos_act)):
		try:
			pos_act[j] = pos_act[j][:,cols] #extracting the needed column
		except IndexError:
			#print(j) 
			pos_act[j]= np.stack(pos_act[j][:-1])
			pos_act[j] = pos_act[j][:,cols] #extracting the needed column
	return pos_act
	

#here we take the torso as the origin of coordinates
def center_torso(activities):
	#Positional activities only
	l=[[] for i in range(len(activities))]
	for i in range(len(activities)):
		for j in range(len(activities[i])):
			torso=activities[i][j][2]
			l[i].append([torso for i in range(15)])
	norm_activities= np.array(activities)-np.array(l)
	return norm_activities

def index_joint(joint_str):
	"""This function is for data composed of only positional information"""

	dic = {"head":0,"neck":1,"torso":2,"left_shoulder":3,"left_elbow":4,"right_shoulder":5,"right_elbow":6,"left_hip":7,"left_knee":8,"right_hip":9,"right_knee":10,"left_hand":11,"right_hand":12,"left_foot":13,"right_foot":14}
	index=dic[joint_str]
	return index

"""
def index_joint_complete_data(joint_str):
	dic = {"head":1,"neck":2,"torso":3,"left_shoulder":4,"left_elbow":5,"right_shoulder":6,"right_elbow":7,"left_hip":8,"left_knee":9,"right_hip":10,"right_knee":11,"left_hand":12,"right_hand":13,"left_foot":14,"right_foot":15}
	index = dic[joint_str]*14
"""

#returns euclidean distance
def joint_dist(activity,joint1,joint2,frame):
	""" We are assuming that the activity has only position of joints"""
	joint1_3d = activity[frame][index_joint(joint1)]
	joint2_3d = activity[frame][index_joint(joint2)]
	dist = distance.euclidean(joint1_3d,joint2_3d)
	return dist

#returns 3D distance
def joint_dist_3d(activity,joint1,joint2,frame):
	""" We are assuming that the activity has only position of joints"""
	joint1_3d = activity[frame][index_joint(joint1)]
	joint2_3d = activity[frame][index_joint(joint2)]
	dist = joint1_3d-joint2_3d
	return dist

#returns 3D distance. Input is only one activity
def joint_mvnt(activity, joint, frame_t1, frame_t0=0):
	""" Calcultes the joint movement in x,y,z i.e. the distance between two frames of a single joint"""
	joint_t0 = activity[frame_t0][index_joint(joint)]
	joint_t1 = activity[frame_t1][index_joint(joint)]
	#dist = joint_t1-joint_t0
	dist = [joint_t1[0]-joint_t0[0],joint_t1[1]-joint_t0[1]]
	return dist

def normalize_by_height(activity):

	'''
	Assumed one activity with only positional coordinates

	normalize skeleton by the "height" of a person in the frame, where height is dist(foot, knee) + dist(knee, hip) + dist(torso, neck) + dist(neck, head)
	activity.shape = (n_frames,n_joints,3)
	'''
	normed_activity = np.zeros(activity.shape)
	for frame_num in range(activity.shape[0]):
		foot_knee = np.linalg.norm(joint_dist(activity,"right_foot","right_knee",frame_num))
		knee_hip = np.linalg.norm(joint_dist(activity,"right_knee","right_hip",frame_num))
		torso_neck = np.linalg.norm(joint_dist(activity,"torso","neck",frame_num))
		neck_head = np.linalg.norm(joint_dist(activity,"neck","head",frame_num))
		height = foot_knee + knee_hip + torso_neck + neck_head
		normed_activity[frame_num] = np.array(activity[frame_num]) / height #is this done iteratively to each entry of matrix?
	return normed_activity

def normalize_by_height_all(activities):
	normed_activities = []
	for activity in activities:
		normed = normalize_by_height(activity)
		normed_activities.append(normed)
	return normed_activities

def standardize_features(features):
	sc = StandardScaler()
	return sc.fit(features).transform(features)


def frame_feature_vector(activity,frame):
	#vector of 14. 19 in my case, I think might be important for some differentiations, such as drinking water and talking on phone
	r_handhead = joint_dist(activity,"head","right_hand",frame)
	l_handhead = joint_dist(activity,"head","left_hand",frame)
	r_shoufoot = joint_dist(activity,"right_shoulder","right_foot",frame)
	l_shoufoot = joint_dist(activity,"left_shoulder","left_foot",frame)
	r_feethip = joint_dist(activity,"right_foot","right_hip",frame)
	l_feethip = joint_dist(activity,"left_foot","left_hip",frame)
	
	
	
	f1 = math.sqrt(r_handhead+l_handhead) #distance of both hands to the head
	f2 = joint_dist(activity,"right_hand","left_hand",frame) #distance between hands
	f3 = math.sqrt(r_shoufoot**2+l_shoufoot**2) #distance between shoulders and hips
	f4 = math.sqrt(r_feethip**2+l_feethip**2) #distance between feet and hip
	f5 = joint_mvnt(activity,"right_hand",frame,frame-1) 
	f6 = joint_mvnt(activity,"left_hand",frame,frame-1)
	f7 = joint_mvnt(activity,"right_elbow",frame,frame-1)
	f8 = joint_mvnt(activity,"left_elbow",frame,frame-1)
	f9 = joint_mvnt(activity,"head",frame,frame-1)
	features = np.array([f1,f2,f3,f4,*f5,*f6,*f7,*f8,*f9]).flatten()
	#comments to this: 
	#1. one interesting feature would be torse inclination. 
	#2. do you think elbow movement carries a lot of information for us?
	return features

def activity_feature_vector(activity,dim = 14):
	#Devuelve un feature vector de toda la actividad.
	n_frames = len(activity)
	features_frames = [frame_feature_vector(activity,frame) for frame in range(n_frames)] #array of features for all frames in the activity
	kynetic_energy = []
	for k in features_frames:
		ke = np.linalg.norm(k[4])+np.linalg.norm(k[5])
		kynetic_energy.append(ke)
	ke_sorted = np.argsort(kynetic_energy)
	b = np.array([0,len(activity)-1])
	ke_sorted = np.setdiff1d(ke_sorted,b,True)
	for k in range(len(kynetic_energy)):
		if ke_sorted[k+1]<ke_sorted[k]:
			max_index = k
			break
		else:
			continue
	#ke_interest_index = [0,*ke_sorted[:max_index+1]]
	ke_interest_index = np.sort([0,*ke_sorted[:dim-2]]) 
	ke_interest_index = [*ke_interest_index,-1]
	features = np.array(features_frames)[ke_interest_index]
	#return [ke_interest_index,features]
	return features

def activities_feature_vector(activities,dim = -1):
	if dim == -1:
		activities_features = [activity_feature_vector(activity,dim = len(activity)) for activity in activities]
	else:
		activities_features = [activity_feature_vector(activity,dim) for activity in activities]
	return activities_features

def partition_activity(activity_complete_feature,divisor):
	#input is dim_features X n_frames
	parted_activity = []
	for i in range(len(activity_complete_feature)-divisor):
		parted_activity.append(activity_complete_feature[i:i+divisor])
	return np.array(parted_activity) #output is dim_features X divisor X (n_frames-divisor)

def partition_activities(activities_complete_features,divisor):
	parted_activities= [partition_activity(activities_complete_features[i],divisor) for i in range(len(activities_complete_features))]
	return parted_activities #output is dim_features X divisor X (n_frames-divisor)

def multiplicate_labels(parted_activities,labels):
	multiple_labels = []
	for i in range(len(parted_activities)):
		for j in range(len(parted_activities[i])):
			#print(i)
			multiple_labels.append(labels[i])
	return multiple_labels


	
'''
def PCA(features):


	#extract features one: velocities and that.
	#extraction2: PCA analysis



	embedding=umap.UMAP(n_components=2,n_neighbors=25,spread=2,metric=metric,verbose=True)
	H = embedding.fit_transform(data,y=cols)
	fig = plt.figure()
	plt.scatter(H[:,0],H[:,1],c=cols,s=5)
	plt.show()
'''	

