import torch
import numpy as np

# {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_keypoints' : centres}

class ZeroCenter(object):
    def __init__(self, mean : np.array = None):
        self.mean = mean

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        skeletons = sample['key_points'] 
        
        if self.mean is not None:
            point_clouds_centered = point_clouds - self.mean
            skeletons_centered = skeletons - self.mean
        else:
            mean = np.mean(point_clouds, axis=(1, 2))
            point_clouds_centered = point_clouds - mean[:, np.newaxis, np.newaxis, :]
            skeletons_centered = skeletons - mean[:, np.newaxis, :]
        
        sample['sequences'] = point_clouds_centered
        sample['key_points'] = skeletons_centered
        sample['root_key_points'] = skeletons_centered[:, 0, :].copy()
        return sample
    
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, torch.Tensor]:
        point_clouds = np.moveaxis(sample['sequences'], 3, 1)
        skeletons = np.moveaxis(sample['key_points'], 2, 1) 
        
        sample['sequences'] = torch.from_numpy(np.float32(point_clouds))
        sample['valid_points'] = torch.from_numpy(np.float32(sample['valid_points']))
        sample['key_points'] = torch.from_numpy(np.float32(skeletons))
        sample['root_key_points'] = torch.from_numpy(np.float32(sample['root_key_points']))
        if 'segmentations' in sample:
            sample['segmentations'] = torch.from_numpy(np.float32(sample['segmentations']))
        return sample
    
class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, np.array]:
        point_clouds = torch.movedim(sample['sequences'], 1, 3)
        skeletons = torch.movedim(sample['key_points'], 1, 2) 
        
        sample['sequences'] = point_clouds.numpy()
        sample['valid_points'] = sample['valid_points'].numpy()
        sample['key_points'] = skeletons.numpy()
        sample['root_key_points'] = sample['root_key_points'].numpy()
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'].numpy()
        return sample

class Rescale(object):
    def __init__(self, min_ : np.array, max_ : np.array, a : int = -1, b : int = 1):
        self.min = min_
        self.max = max_
        self.a = a
        self.b = b

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        skeletons = sample['key_points'] 
        
        point_clouds_scaled = self.a + ((point_clouds - self.min) * (self.b - self.a) / (self.max - self.min))
        skeletons_scaled = self.a + ((skeletons - self.min) * (self.b - self.a) / (self.max - self.min))
        
        sample['sequences'] = point_clouds_scaled
        sample['key_points'] = skeletons_scaled
        sample['root_key_points'] = skeletons_scaled[:, 0, :].copy()
        return sample

class RandomCrop(object):
    def __init__(self, output_shape : tuple[int, int], input_shape : tuple[int, int] = None):
        self.output_shape = output_shape
        if input_shape is None:
            self.high_w = None
            self.high_h = None
        else:
            h0, w0 = input_shape
            h1, w1, = output_shape
            self.high_w = w0 - w1
            self.high_h = h0 - h1      

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        masks = sample['valid_points']
        if 'segmentations' in sample:
            segmentations = sample['segmentations']
        h1, w1, = self.output_shape   

        if self.high_w is None or self.high_h is None:
            _, h1, w1, _ = point_clouds.size()
        else:
            high_w = self.high_w
            high_h = self.high_h
        offset_x = np.random.randint(0, high_w)
        offset_y = np.random.randint(0, high_h)
        cropped_point_clouds = point_clouds[:, offset_y : offset_y + h1, offset_x : offset_x + w1, :]
        cropped_masks = masks[:, offset_y : offset_y + h1, offset_x : offset_x + w1]
        if 'segmentations' in sample:
            cropped_segmentations = segmentations[:, offset_y : offset_y + h1, offset_x : offset_x + w1]
        
        sample['sequences'] = cropped_point_clouds
        sample['valid_points'] = cropped_masks
        if 'segmentations' in sample:
            sample['segmentations'] = cropped_segmentations
        return sample

class ZeroOutEntries(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences'] 
        masks = sample['valid_points']

        point_clouds[masks == False] = np.zeros(3)

        sample['sequences'] = point_clouds     
        return sample
    
class RelativeJointsPosition(object):
    def __init__(self):
        self.joints_id = [11, 8, 5, 2, # their right leg
                          10, 7, 4, 1, # their left leg
                          0, 3, 6, 9, 12, 13, # spine
                          15, 17, 19,  # their right arm
                          14, 16, 18] # their left arm
        self.parents_id= [8, 5, 2, 0, 
                          7, 4, 1, 0,
                          0, 0, 3, 6, 9, 12,
                          12, 15, 17,
                          12, 14, 16]
        
    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        skeletons = sample['key_points'] 

        skeletons[:, self.joints_id, :] = skeletons[:, self.joints_id, :] - skeletons[:, self.parents_id, :]

        sample['key_points'] = skeletons
        return sample
        

class ToDevice(object):
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample['sequences'] = sample['sequences'].to(device=self.device)
        sample['valid_points'] = sample['valid_points'].to(device=self.device)
        sample['key_points'] = sample['key_points'].to(device=self.device)
        sample['root_key_points'] = sample['root_key_points'].to(device=self.device)
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'].to(device=self.device)

        return sample


class AutoGrad(object):
    def __init__(self, requires_grad=True):
        self.grad = requires_grad

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample['sequences'] = sample['sequences'].requires_grad_(self.grad)
        #sample['valid_points'] = sample['valid_points'].requires_grad_(self.grad)
        sample['key_points'] = sample['key_points'].requires_grad_(self.grad)
        sample['root_key_points'] = sample['root_key_points'].requires_grad_(self.grad)
        #if 'segmentations' in sample:
            #sample['segmentations'] = sample['segmentations'].requires_grad_(self.grad)

        return sample
    
class Shuffle(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        s, h, w, c = sample['sequences'].shape
        idx = np.random.choice(s, s, replace=False)
        sample['sequences'] = sample['sequences'][idx]
        sample['valid_points'] = sample['valid_points'][idx]
        sample['key_points'] = sample['key_points'][idx]
        sample['root_key_points'] = sample['root_key_points'][idx]
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'][idx]

        return sample
    


