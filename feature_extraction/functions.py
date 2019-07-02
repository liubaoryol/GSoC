import numba
from scipy.spatial import distance
import umap

@numba.njit()
def metric(act1,act2):
	dist=0
	min_dim = min(len(act1),len(act2))
	for i in range(min_dim):
		for j in range(len(act1[i])):
			dist +=distance.euclidean(act1[i][j],act2[i][j])
	return dist


#here we take the torso as the origin of coordinates
def center_torso(activities):
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

def joint_dist(activity,join1,joint2,frame):
""" We are assuming that the activity has only position of joints"""
	joint1_3d = activity[frame][index_joint(joint1)]
	joint2_3d = activity[frame][index_joint(joint2)]
	dist = distance.euclidean(joint1_3d,joint2_3d)
	return dist


def joint_mvnt(activity, joint, frame_t1, frame_t0=0)
""" Calcultes the joint movement, i.e. the distance between two frames of a single joint"""
	joint_t0 = activity[frame_t0][index_joint(joint)]
	joint_t1 = activity[frame_t1][index_joint(joint)]
	dist = joint_t1-joint_t0
	return dist

def normalize_by_height(activity):

    '''
    Assumed activity with only positional coordinates
    
    normalize skeleton by the "height" of a person in the frame, where height is dist(foot, knee) + dist(knee, hip) + dist(torso, neck) + dist(neck, head)
    activity.shape = (n_frames,n_joints,3)
    '''
	normed_activity = np.zeros((activity.shape))
	for frame_num in range(activity.shape[0]):
		foot_knee = joint_dist(activity,"right_foot","right_knee",frame_num)
		knee_hip = joint_dist(activity,"right_knee","right_hip",frame_num)
		torso_neck = joint_dist(activity,"torso","neck",frame_num)
		neck_head = joint_dist(activity,"neck","head",frame_num)
		height = foot_knee + knee_hip + torso_neck + neck_head
		normed_activity[i] = normed_activity[i] / height #is this done iteratively to each entry of matrix?
	return normed_activity



def standardize_features(features):
	sc = StandardScaler()
	return sc.fit(features).transform(features)


def features_vector(activity,frame):
	#vector of 14. 19 in my case, I think might be important for some differentiations, such as drinking water and talking on phone
	r_handhead = joint_dist(activity,"head","right_hand",frame)
	l_handhead = joint_dist(activity,"head","left_hand",frame)
	r_shouhip = joint_dist(activity,"right_shoulder","right_hip",frame)
	l_shouhip = joint_dist(activity,"left_shoulder","left_hip",frame)
	r_feethip = joint_dist(activity,"right_foot","right_hip",frame)
	l_feethip = joint_dist(activity,"left_foot","left_hip",frame)
	
	
	
	f1 = math.sqrt(r_handhead**2+l_handhead**2)
	f2 = joint_dist(activity,"right_hand","left_hand",frame)
	f3 = math.sqrt(r_shouhip**2+l_shouhip**2)
	f4 = math.sqrt(r_shouhip**2+l_shouhip**2)
	f6 = joint_mvnt(activity,"right_hand",frame,frame-1)
	f7 = joint_mvnt(activity,"right_hand",frame,frame-1)
	f8 = joint_mvnt(activity,"right_hand",frame,frame-1)
	f9 = joint_mvnt(activity,"right_hand",frame,frame-1)
	features = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9]).flatten()
	return features



def PCA(features):


#extract features one: velocities and that.
#extraction2: PCA analysis



embedding=umap.UMAP(n_components=2,n_neighbors=25,spread=2,metric=metric,verbose=True)
H = embedding.fit_transform(data,y=cols)
fig = plt.figure()
plt.scatter(H[:,0],H[:,1],c=cols,s=5)
plt.show()

		

