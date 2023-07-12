import torch
import os 

from Networks import PoseEstimation3DNetwork 
from Dataset import *
from Transforms import *
from Trainer  import *
from utils import *


if __name__ == '__main__':
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    transforms = transforms.Compose([ZeroCenter(),
                                     Rescale(min_, max_, -1, 1),
                                     RandomCrop((224, 224), (257, 344)),
                                     RelativeJointsPosition(),
                                     ZeroOutEntries(),
                                     ToTensor(),
                                     ToDevice(device)])

    ACCAD_ts_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='testing', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=False)
    
    CMU_ts_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='testing',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=False)
    
    EKUT_ts_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='testing',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=False)
    
    Eyes_Japan_ts_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='testing',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=False)
    
    model_path = os.path.join('models', 'experiment_3', 'net_22.pt')
    model = PoseEstimation3DNetwork.NetworkBatchNorm(ACCAD_ts_dataset.joints)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    sample_idx = 7
    sample = CMU_ts_dataset[sample_idx] #Eyes_Japan_ts_dataset[idx]
    predictions = model(sample).detach().cpu()

    idx = 10
    predictions = np.moveaxis(predictions.numpy(), 1, 2)
    sequences = sample['sequences'].detach().cpu()
    sequences = np.moveaxis(sequences.numpy(), 1, 3)
    targets = sample['key_points'].detach().cpu()
    targets = np.moveaxis(targets.numpy(), 1, 2)
    centres = sample['root_key_points'].detach().cpu()

    target_skeleton = reconstruct_skeleton(targets[idx], centres[idx])
    predicted_skeleton = reconstruct_skeleton(predictions[idx], centres[idx])
    ax = plot_body(sequences[idx])
    #plot_skeleton(targets[idx], ax)
    plot_skeleton(target_skeleton, ax)
    
    ax = plot_body(sequences[idx])
    plot_skeleton(predicted_skeleton, ax)
    #plot_skeleton(predictions[idx], ax)
    plt.show()

    
