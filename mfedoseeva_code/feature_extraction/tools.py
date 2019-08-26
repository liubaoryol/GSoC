import numpy as np

def normalize(X, valid_frame_num):
    '''
    normalize for each feature across all frames, separately for each sample
    in_shape = (N, 14, n_frames)
    '''
    N, feat, T  = X.shape
    out = np.zeros((N, feat, T))
    for i in range(N):
        for f in range(feat):
            max_feat = np.max(np.absolute(X[i, f, :]))
            out[i, f, :] = X[i, f, :] * (1/max_feat)
    return out

def normalize_by_height(feature, X, valid_frame_num):
    '''
    normalize a (distance) feature by the "height" of a person in the frame, where height is calc. as dist(foot, knee) + dist(knee, hip) + dist(torso, neck) +
    dist(neck, head)
    feature.shape = (N, n_frame)
    '''
    N, C, T, V = X.shape
    normed_feature = np.zeros((feature.shape))
    for i in range(N):
        for t in range(valid_frame_num[i]):
            foot_knee = dist_to_torso_singleframe(X[i, :, t, 13], X[i, :, t, 8])
            knee_hip = dist_to_torso_singleframe(X[i, :, t, 8], X[i, :, t, 7])
            torso_neck = dist_to_torso_singleframe(X[i, :, t, 2], X[i, :, t, 1])
            neck_head = dist_to_torso_singleframe(X[i, :, t, 1], X[i, :, t, 0])
            height = foot_knee + knee_hip + torso_neck + neck_head
            normed_feature[i, t] = feature[i, t] / height
    return normed_feature

def center(X, valid_frame_num):
    '''
    origin is moved to torso coordinates for each frame.
    '''
    # c - xyz, t - frames, v - join
    N, C, _, V = X.shape
    for i in range(N):
        for t in range(valid_frame_num[i]):
            # torso is the 3rd joint
            torso_coord = X[i, :, t, 2]
            for v in range(V):
                X[i, :, t, v] -= torso_coord
    return X

def dist_to_torso(joint_data, valid_frame_num):
    '''
    torso coords are always zero since we center relative to torso.
    in_shape = (3, n_frames)
    out_shape = (n_frames)
    '''
    dist_data = np.zeros(valid_frame_num)
    for t in range(valid_frame_num):
        dist_data[t] = np.linalg.norm(joint_data[:, t])   
    return dist_data


def dist_to_torso_allsamples(joint_data, valid_frame_num):
    '''
    torso coords are always zero since we center relative to torso.
    in_shape = (N, 3, n_frames)
    out_shape = (N, n_frames)
    '''
    N, _, T = joint_data.shape
    dist_data = np.zeros((N, T))
    for i in range(N):
        for t in range(valid_frame_num[i]):
            dist_data[i, t] = np.linalg.norm(joint_data[i, :, t])   
    return dist_data


def dist_to_torso_singleframe(joint_data):
    '''
    assumes torso coords to be zero
    dist_to_torso for single frame
    '''
    return np.linalg.norm(joint_data)


def dist_to_joint(joint1, joint2, valid_frame_num1, valid_frame_num2):
    '''
    in_shape = (3, n_frames) for both
    out_shape = (n_frames)
    '''
    assert(valid_frame_num1 == valid_frame_num2)
    dist_data = np.zeros(valid_frame_num1)
    for t in range(valid_frame_num1):
        dist_data[t] = np.linalg.norm(joint1[:, t] - joint2[:, t])
    return dist_data


def dist_to_joint_allsamples(joint1, joint2, valid_frame_num):
    '''
    in_shape = (N, 3, n_frames) for both
    out_shape = (N, n_frames)
    '''
    N, _, T = joint1.shape
    dist_data = np.zeros((N, T))
    for i in range(N):
        for t in range(valid_frame_num[i]):
            dist_data[i, t] = np.linalg.norm(joint1[i, :, t] - joint2[i, :, t])
    return dist_data

def dist_to_joint_single(joint1, joint2):
    '''
    dist_to_torso for single frame
    '''
    return np.linalg.norm(joint1 - joint2)


def horizontal_flip(X, valid_frame_num):
    '''
    flip horizontally all data relative to torso, assumes that data is centered at torso
    in_shape = out_shape
    '''
    for t in range(valid_frame_num):
        X[0, t, :] *= -1

def clip_samples_even(X, valid_frames_num):
    '''
    make sequnces the same length equal to the shortest sequence
    '''
    min_seq = min(valid_frames_num)
    return X[:, :, :min_seq, :], min_seq

def diff_position_x(X, num_frames):
    '''
    in_shape of joint (N, 3, n_frames)
    out_shape (N, n_frames)
    '''
    N, _, T = X.shape
    out = np.zeros((N, T))
    for i in range(N):
        init_x = X[i, 0, 0]
        for t in range(num_frames[i]):
            out[i, t] = X[i, 0, t] - init_x
    return out

def diff_position_y(X, num_frames):
    '''
    in_shape of joint (N, 3, n_frames)
    out_shape (N, n_frames)
    '''
    N, _, T = X.shape
    out = np.zeros((N, T))
    for i in range(N):
        init_y = X[i, 1, 0]
        for t in range(num_frames[i]):
            out[i, t] = X[i, 1, t] - init_y
    return out


def frame_by_frame_samples(X, valid_frame_num):
    '''
    turns every frame into a sample
    in_shape (n_samples, features, n_frames)
    out_shape (sum(valid_frame_num), features)
    '''
    N, F, _ = X.shape
    total_samples = sum(valid_frame_num)
    X_new = np.zeros((total_samples, F))
    prev = 0
    for i in range(N):
        for j in range(valid_frame_num[i]):
            X_new[prev + j] = X[i, :, j]
        prev += valid_frame_num[i]
    return X_new

def frame_by_frame_labels(labels, valid_frame_num):
    '''
    produces labels for each frame
    in-len = n_samples
    out_len = sum(valid_frame_num)
    '''
    new_labels = []
    for i in range(len(labels)):
        for j in range(valid_frame_num[i]):
            new_labels.append(labels[i])
    return new_labels


def flatten(X, valid_frame_num):
    '''
    turn two-dim data into one-dim data by stacking
    in_shape = (num_frames, features)
    out_shape = (valid_frames*features)
    '''
    _, V = X.shape
    data = np.zeros(valid_frame_num * V)
    for i in range(valid_frame_num):
        data[i*valid_frame_num : i*valid_frame_num + valid_frame_num] = X[i]
    return data


# def cut_sample(X, valid_frame_num, length):
#     '''
#     takes in data and returns valid_frame_num/length samples
#     in_shape = (xyz, n_frames, joints)
#     '''
#     times = int(valid_frame_num/length)
#     samples = []
#     for i in range(times):
#         idx_s = i * length
#         idx_e = i * length + length
#         s = X[:, idx_s:idx_e, :]
#         samples.append(s)
#     return samples, times

# def sample_frames_from_sequnce(X):
#     '''
#     assumes sample is cut already
#     in_shape = (num_frames, features)
#     takes two-dim data and reduced the frames
#     out_shape = (9, features)
#     '''
#     T, V = X.shape
#     # for now implemented for 45
#     assert(T == 45)
#     idx = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44]
#     data = np.zeroes((9, V))
#     for i in range(9):
#         data[i] = X[idx[i], :]

# def cos_dist(joint1, joint2, valid_frame_num1, valid_frame_num2):
#     '''
#     cosine distance of a joint relative to another joint
#     in_shape = (3, n_frames)
#     out_shape = (3, n_frames)
#     '''
#     assert(valid_frame_num1 == valid_frame_num2)
    
#     out = np.zeros(valid_frame_num1)
#     for t in range(valid_frame_num1):
#         dot_prod = np.dot(joint1[t], joint2[t])
#         a = np.linalg.norm(joint1[:,t])
#         b = np.linalg.norm(joint2[:, t])
#         out[t] = dot_prod/(a * b)   
#     return out

# def cos_dist_totorso(joint1, torso_coords, valid_frame_num):
#     '''
#     cosine distance of a joint relative to torso, requires non-centered data for both joints
#     in_shape = (N, 3, n_frames)
#     out_shape = (N, 3, n_frames)
#     '''
    
#     N, _, T = joint1.shape
#     out = np.zeros((N, T))
#     for i in range(N):
#         for t in range(valid_frame_num[i]):
#             dot_prod = np.dot(joint1[i, :, t], torso_coords[i, :, t])
#             print(dot_prod)
#             a = np.linalg.norm(joint1[i, :,t])
#             b = np.linalg.norm(torso_coords[i, :, t])
#             print(a*b)
#             out[t] = dot_prod/(a * b)   
#     return out