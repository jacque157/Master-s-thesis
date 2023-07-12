import numpy as np
import os
import re
import scipy
import ntpath

MAT_ROOT = os.path.join('D:', 'škola', 'matfyz', 'mgr', 'diplomovka', 'dataset', 'mat') # Path to .mat files of point clouds
DATASETS = ['EKUT', 'Eyes_Japan', 'ACCAD', 'CMU']

SKELETONS_ROOT = os.path.join('D:', 'škola', 'matfyz', 'mgr', 'diplomovka', 'dataset') # Path to .obj files of skeletons
MASK = np.zeros(52)
MASK[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]] = 1  # skeleton key points used  
MASK = MASK == 1

OUT_ROOT = os.path.join('.')


def read_key_points(path):
    vertices = []
    with open(path) as file:
        for line in file:
            if line[0] == 'v':
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append(np.array([x, y, z]))
    return np.array(vertices)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def create_point_cloud(file_path, dataset_path):
    mat = scipy.io.loadmat(file_path)
    structured_point_cloud = mat['img']

    point_cloud = structured_point_cloud.reshape(-1, 3)
    store_zero_centered_min_max(dataset_path, point_cloud)
    
    file_name = path_leaf(file_path)
    sequence, body, camera = map(int, re.findall(r'\d+', file_name))
    if 'female' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'female',
                                  f'sequence_{sequence}', 
                                  f'camera_{camera}')
    elif 'male' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'male',
                                  f'sequence_{sequence}', 
                                  f'camera_{camera}')
    else:
        out_folder = os.path.join(dataset_path, 
                                f'sequence_{sequence}', 
                                f'camera_{camera}')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_name = f'pose_{body}'
    out_path = os.path.join(out_folder, out_name)
    np.save(out_path, structured_point_cloud)

def store_zero_centered_min_max(dataset_path, point_cloud):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    mask = np.logical_or(np.logical_or(x != 0, y != 0), z != 0)
    points_of_interest = point_cloud[mask]
    points_of_interest -= np.mean(points_of_interest, axis=0)
    min_ = np.min(points_of_interest, axis=0)
    max_ = np.max(points_of_interest, axis=0)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    file = os.path.join(dataset_path, 'min_max.npy')
    if os.path.exists(file):
        min_max = np.load(file)
        current_min = min_max[0,:]
        current_max = min_max[1,:]
        min_ = np.min(np.stack([min_, current_min]), axis=0)
        max_ = np.max(np.stack([max_, current_max]), axis=0)
        
    min_max = np.stack([min_, max_])
    np.save(file, min_max)

def create_point_clouds(poses_folder, root):
    for file_name in os.listdir(poses_folder):
        file_path = os.path.join(poses_folder, file_name)
        if os.path.isfile(file_path):
            create_point_cloud(file_path, root)
        else:
            create_point_clouds(file_path, root)

def create_key_points(poses_folder, root, mask):
    for file_name in os.listdir(poses_folder):
        file_path = os.path.join(poses_folder, file_name)
        if os.path.isfile(file_path):
            if 'pose' in file_path:
                create_skeleton(file_path, root, mask)
        else:
            create_key_points(file_path, root, mask)
        
def create_skeleton(file_path, dataset_path, mask):
    key_points_all = read_key_points(file_path)
    key_points = key_points_all[mask]

    file_name = path_leaf(file_path)
    sequence, body = map(int, re.findall(r'\d+', file_name))
    if 'female' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'female',
                                  f'sequence_{sequence}',
                                  'skeletons')
    elif 'male' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'male',
                                  f'sequence_{sequence}',
                                  'skeletons')
    else:
        out_folder = os.path.join(dataset_path, 
                                f'sequence_{sequence}',
                                  'skeletons')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_name = f'pose_{body}_skeleton'
    out_path = os.path.join(out_folder, out_name)
    np.save(out_path, key_points)


def enum_sequences(poses_folder):
    for sequence_folder in os.listdir(poses_folder):
        folder_path = os.path.join(poses_folder, sequence_folder)
        if os.path.isdir(folder_path):
            enum_sequence(folder_path)

def annotate(pose, skeleton):
    difference = pose[:, :, None, :] - skeleton[None, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    annotation = np.argmin(distances, axis=2)

    ground_mask = np.abs(pose[:,:,2]) <= 10
    annotation[ground_mask] = len(skeleton)
    
    return annotation

def enum_sequence(folder_path):
    poses_dir = os.path.join(folder_path, 'camera_1') 
    number_of_poses = np.uint16([len(os.listdir(poses_dir))])
    out_path = os.path.join(folder_path, 'sequence_length')
    np.save(out_path, number_of_poses)

def annotate_sequences(folder_path):
    for sequence_name in os.listdir(folder_path):
        sequence_path = os.path.join(folder_path, sequence_name)
        annotate_sequence(sequence_path)

def annotate_sequence(folder_path):   
    number_of_sequnces = len(os.listdir(os.path.join(folder_path, 'camera_1')))
    
    for i in range(number_of_sequnces):
        skeleton_path = os.path.join(folder_path, 'skeletons', f'pose_{i}_skeleton.npy')
        skeleton = np.load(skeleton_path)

        
        for j in range(1, 5):
            camera_path = os.path.join(folder_path, f'camera_{j}')
            pose_path = os.path.join(camera_path, f'pose_{i}.npy')
            pose = np.load(pose_path)
            annotations_path = os.path.join(folder_path, 'segmentations', f'camera_{j}') 
            if not os.path.exists(annotations_path):
                os.makedirs(annotations_path)
            annotation = annotate(pose, skeleton)
            annotation_path = os.path.join(annotations_path, f'pose_{i}_annotation.npy')
            np.save(annotation_path, annotation)



if __name__ == '__main__':
    for dataset in DATASETS:
        dataset_path = os.path.join(MAT_ROOT, dataset)

        male_dataset_path = os.path.join(dataset_path, 'male')    
        if os.path.exists(male_dataset_path):
            create_point_clouds(male_dataset_path, os.path.join(OUT_ROOT, dataset))
            enum_sequences(os.path.join(OUT_ROOT, dataset, 'male'))

        female_dataset_path = os.path.join(dataset_path, 'female')
        if os.path.exists(female_dataset_path):
            create_point_clouds(female_dataset_path, os.path.join(OUT_ROOT, dataset))
            enum_sequences(os.path.join(OUT_ROOT, dataset, 'female'))
        
        skeletons_path = os.path.join(SKELETONS_ROOT, dataset)

        male_dataset_path = os.path.join(skeletons_path, 'male')    
        if os.path.exists(male_dataset_path):
            create_key_points(male_dataset_path, os.path.join(OUT_ROOT, dataset), MASK)

        female_dataset_path = os.path.join(skeletons_path, 'female')
        if os.path.exists(female_dataset_path):
            create_key_points(female_dataset_path, os.path.join(OUT_ROOT, dataset), MASK)

        dataset_male_path = os.path.join(OUT_ROOT, dataset, 'male')
        if os.path.exists(dataset_male_path):
            annotate_sequences(dataset_male_path)

        dataset_female_path = os.path.join(OUT_ROOT, dataset, 'female')
        if os.path.exists(dataset_female_path):
            annotate_sequences(dataset_female_path)


