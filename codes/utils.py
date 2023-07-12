import numpy as np
import os 
import matplotlib.pyplot as plt

def find_global_min_max(root, names):
    best_min = np.array([float('inf'), float('inf'), float('inf')])
    best_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    for name in names:
        path = os.path.join(root, name, 'min_max.npy')
        min_, max_ = np.load(path)
        mask = min_ < best_min
        best_min[mask] = min_[mask]

        mask = max_ > best_max
        best_max[mask] = max_[mask]
    return best_min, best_max

def find_projection_matrix(img):
    x = img[:,:,0]
    y = img[:,:,1]
    z = img[:,:,2]

    mask = np.logical_or(np.logical_or(x != 0, y != 0), z != 0)
    points = img[mask]    
    h, w, x = img.shape
    vu = np.mgrid[0:h, 0:w]

    px = vu[1][mask]
    py = vu[0][mask]

    sys = []
    for i in range(0, len(px), 10):
        x, y, z = points[i]
        u, v = px[i], py[i]
        sys.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        sys.append([0, 0, 0, 0, x, y, z, 1,  -v * x, -v * y, -v * z, -v])

    U, s, V = np.linalg.svd(sys)

    M = V[-1, :].reshape(3, 4)
    return M

def plot_body(img, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pt_cloud = img.reshape(-1, 3)
    pt_cloud = np.random.permutation(pt_cloud)[::10, :]
    ax.scatter(pt_cloud[:, 0], 
               pt_cloud[:, 1], 
               pt_cloud[:, 2], 
               marker='o', 
               color='blue', 
               alpha=0.1)
    return ax
    
def plot_skeleton(pose, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pairs = [(11, 8), (8, 5), (5, 2), (2, 0), # their right leg
             (10, 7), (7, 4), (4, 1), (1, 0), # their left leg
             (0, 3), (3, 6), (6, 9), (9, 12), (12, 13), # spine
             (19, 17), (17, 15), (15, 12), # their right arm
             (18, 16), (16, 14), (14, 12)] # their left arm
    
    for (start, end) in pairs:
        x_start, y_start, z_start = pose[start, :]
        x_end, y_end, z_end = pose[end, :]
        ax.plot((x_start, x_end),
                (y_start, y_end),
                (z_start, z_end),
                marker='x', 
                color='red')
        
    for i, v in enumerate(pose):
        label = f'{i}' 
        ax.text(v[0], v[1], v[2], label)

    return ax

def load_body(root, dataset, seq, frame, camera):
    pose_path = os.path.join(root,
                             dataset,
                             'male',
                             f'sequence_{seq}',
                             f'camera_{camera}',
                             f'pose_{frame}.npy')
    if os.path.exists(pose_path):
        return np.load(pose_path)
    
    pose_path = os.path.join(root,
                             dataset,
                             'female',
                             f'sequence_{seq}',
                             f'camera_{camera}',
                             f'pose_{frame}.npy')
    if os.path.exists(pose_path):
        return np.load(pose_path)

def load_skeleton(root, dataset, seq, frame):
    skeleton_path = os.path.join(root,
                                 dataset,
                                 'male',
                                 f'sequence_{seq}',
                                 'skeletons',
                                 f'pose_{frame}_skeleton.npy')
    if os.path.exists(skeleton_path):
        return np.load(skeleton_path)
    
    skeleton_path = os.path.join(root,
                                 dataset,
                                 'female',
                                 f'sequence_{seq}',
                                 'skeletons',
                                 f'pose_{frame}_skeleton.npy')
    if os.path.exists(skeleton_path):
        return np.load(skeleton_path)

def load_number_of_frames(root, dataset, seq):
    length_path = os.path.join(root,
                               dataset,
                               'female',
                               f'sequence_{seq}',
                               'sequence_length.npy')
    if os.path.exists(length_path):
        return np.load(length_path)
    
    length_path = os.path.join(root,
                               dataset,
                               'male',
                               f'sequence_{seq}',
                               'sequence_length.npy')
    if os.path.exists(length_path):
        return np.load(length_path)
    
def load_segmentation(root, dataset, seq, frame, camera):
    segmentation_path = os.path.join(root,
                                    dataset,
                                    'male',
                                    f'sequence_{seq}',
                                    'segmentations',
                                    f'camera_{camera}',
                                    f'pose_{frame}_annotation.npy')
    if os.path.exists(segmentation_path):
        return np.load(segmentation_path)
    
    segmentation_path = os.path.join(root,
                                    dataset,
                                    'female',
                                    f'sequence_{seq}',
                                    'segmentations',
                                    f'camera_{camera}',
                                    f'pose_{frame}_annotation.npy')
    if os.path.exists(segmentation_path):
        return np.load(segmentation_path)

def reconstruct_skeleton(relative_joints, centre=None):
    skeleton = np.zeros(relative_joints.shape)
    skeleton[0] = np.zeros(3) if centre is None else centre
    skeleton[1] = skeleton[0] + relative_joints[1]
    skeleton[2] = skeleton[0] + relative_joints[2]
    skeleton[3] = skeleton[0] + relative_joints[3]
    skeleton[6] = skeleton[3] + relative_joints[6]
    skeleton[9] = skeleton[6] + relative_joints[9]
    skeleton[12] = skeleton[9] + relative_joints[12]
    skeleton[13] = skeleton[12] + relative_joints[13]

    skeleton[15] = skeleton[12] + relative_joints[15]
    skeleton[17] = skeleton[15] + relative_joints[17]
    skeleton[19] = skeleton[17] + relative_joints[19]

    skeleton[14] = skeleton[12] + relative_joints[14]
    skeleton[16] = skeleton[14] + relative_joints[16]
    skeleton[18] = skeleton[16] + relative_joints[18]

    skeleton[5] = skeleton[2] + relative_joints[5]
    skeleton[8] = skeleton[5] + relative_joints[8]
    skeleton[11] = skeleton[8] + relative_joints[11]

    skeleton[4] = skeleton[1] + relative_joints[4]
    skeleton[7] = skeleton[4] + relative_joints[7]
    skeleton[10] = skeleton[7] + relative_joints[10]
    return skeleton

def annotate(pose, skeleton):
    difference = pose[:, :, None, :] - skeleton[None, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    annotation = np.argmin(distances, axis=2)

    ground_mask = np.abs(pose[:,:,2]) <= 10
    annotation[ground_mask] = len(skeleton)
    
    return annotation

def plot_annotation(pose, annotation, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pt_cloud = pose.reshape(-1, 3)
    pt_cloud_annotation = annotation.reshape(-1, 1)

    pt_cloud = np.concatenate([pt_cloud, pt_cloud_annotation], axis=1)    
    pt_cloud = np.random.permutation(pt_cloud)[::10, :]
    
    for n in range(int(np.max(pt_cloud[:, 3])) + 1):
        mask = pt_cloud[:, 3] == n
        pt_cloud_n = pt_cloud[mask]
        ax.scatter(pt_cloud_n[:, 0], 
                pt_cloud_n[:, 1], 
                pt_cloud_n[:, 2], 
                marker='o', 
                alpha=1)
    return ax

if __name__ == '__main__':
    """min_, max_ = find_global_min_max('Dataset', ['CMU'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['ACCAD'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['EKUT'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['Eyes_Japan'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    print(min_)
    print(max_)
    print()"""

    body = load_body('Dataset', 'CMU', 20, 30, 2)
    ax = plot_body(body)
    skeleton = load_skeleton('Dataset', 'CMU', 20, 30)
    print(len(skeleton))
    plot_skeleton(skeleton, ax)
    #print(load_number_of_frames('Dataset', 'CMU', 20))
    an = annotate(body, skeleton)
    print(an.shape)
    plot_annotation(body, an)
    plt.show()


