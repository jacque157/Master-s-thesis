import os
from typing import Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time


from Transforms import *
from utils import *


class Poses3D(Dataset):
    def __init__(self, path, name, subset=None, transform=ToTensor(), protocol=(0, 1), include_segmentation=True):
        self.path = path
        self.name = name
        self.transform = transform
        self.include_segmentation = include_segmentation
        self.joints = 20
        n = self.get_size()
        indexes = set(range(n))
        k, l = protocol
        validation_indexes = set(range(k % 5, n, 5)) # k, k + 5, k + 10, ...
        testing_indexes = set(range(l % 5, n, 5)) # l, l + 5, l + 10, ...
        training_indexes = indexes - validation_indexes - testing_indexes
        if subset == 'validation':
            self.indexes = validation_indexes
        elif subset == 'testing':
            self.indexes = testing_indexes
        elif subset == 'training':
            self.indexes = training_indexes
        else:
            self.indexes = indexes
        self.indexes = list(self.indexes)

    def get_size(self):
        n = 0
        path = os.path.join(self.path, self.name, 'male')
        if os.path.exists(path):
            n += len(os.listdir(path))

        path = os.path.join(self.path, self.name, 'female')
        if os.path.exists(path):
            n += len(os.listdir(path))

        return n * 4

    def __getitem__(self, index) -> Union[dict[str, np.array], dict[str, torch.Tensor]]:
        index = self.indexes[index]
        camera_index = (index % 4) + 1
        sequence = (index // 4) + 1

        frames = load_number_of_frames(self.path, self.name, sequence)[0]
        if frames is None:
            return None, None, None, None
        
        poses = [load_body(self.path, self.name, sequence, frame, camera_index) for frame in range(frames)]
        skeletons = [load_skeleton(self.path, self.name, sequence, frame) for frame in range(frames)]
        if self.include_segmentation:
            segmentations = [load_segmentation(self.path, self.name, sequence, frame, camera_index) for frame in range(frames)]
            segmentations = np.array(segmentations)
        poses = np.array(poses)
        skeletons = np.array(skeletons)
        

        x = poses[:, :, :, 0]
        y = poses[:, :, :, 1]
        z = poses[:, :, :, 2]
        mask = np.logical_or(np.logical_or(x != 0, y != 0), z != 0)
        centres = skeletons[:, 0, :]

        sample = {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_key_points' : centres}
        if self.include_segmentation:
            sample['segmentations'] = segmentations
        if self.transform:
            sample = self.transform(sample)
        return sample # structured point clouds in sequence, binary masks for valid values (invalid point: (0, 0, 0)), points of skeletons, positions  of root point (pelvis) in skeletons 

    def __len__(self):
        return len(self.indexes)


if __name__ == '__main__':
    n = 0
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    CMU = Poses3D('Dataset', 'ACCAD', 'training', transforms.Compose([ZeroCenter(),
                                                                    Rescale(min_, max_, -1, 1),
                                                                    RandomCrop((224, 224), (257, 344)),
                                                                    RelativeJointsPosition(),
                                                                    ZeroOutEntries(),
                                                                    ToTensor(),
                                                                    ToNumpy()]))

    data = CMU[1]
    seq = data['sequences']
    t = data['key_points']
    c = data['root_key_points']
    an = data['segmentations']
    ax = plot_body(seq[1])
    t = reconstruct_skeleton(t[1], c[1])
    plot_skeleton(t, ax)
    plot_annotation(seq[1], an[1])

    CMU = Poses3D('Dataset', 'ACCAD', 'training', None)
    data = CMU[1]
    seq = data['sequences']
    t = data['key_points']
    c = data['root_key_points']
    an = data['segmentations']
    ax = plot_body(seq[1])
    #t = reconstruct_skeleton(t[0], c[0])
    plot_skeleton(t[1], ax)
    plot_annotation(seq[1], an[1])
    plt.show()
