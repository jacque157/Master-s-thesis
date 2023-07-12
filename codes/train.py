import torch

from Networks import PoseEstimation3DNetwork 
from Dataset import *
from Transforms import *
from Trainer  import *


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    transforms = transforms.Compose([ZeroCenter(),
                                     Rescale(min_, max_, -1, 1),
                                     RandomCrop((224, 224), (257, 344)),
                                     #RelativeJointsPosition(),
                                     ZeroOutEntries(),
                                     Shuffle(),
                                     ToTensor(),
                                     ToDevice(device)])

    ACCAD_tr_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='training', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=False)
    
    ACCAD_val_dataset = Poses3D('Dataset', 
                                'ACCAD', 
                                subset='validation', 
                                transform=transforms,
                                protocol=(0, 1), 
                                include_segmentation=False)
    
    CMU_tr_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='training',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=False)
    
    CMU_val_dataset = Poses3D('Dataset',
                              'CMU',
                              subset='validation',
                              transform=transforms,
                              protocol=(0, 1),
                              include_segmentation=False)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='training',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=False)
    
    EKUT_val_dataset = Poses3D('Dataset',
                               'EKUT',
                               subset='validation',
                               transform=transforms,
                               protocol=(0, 1),
                               include_segmentation=False)
    
    Eyes_Japan_tr_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=False)
    
    Eyes_Japan_val_dataset = Poses3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms,
                                     protocol=(0, 1),
                                     include_segmentation=False)
    
    model = PoseEstimation3DNetwork.NetworkBatchNorm(ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    
    trainer = Trainer(model, [CMU_tr_dataset,
                              ACCAD_tr_dataset, 
                              EKUT_tr_dataset,
                              Eyes_Japan_tr_dataset], 

                              [CMU_val_dataset,
                               ACCAD_val_dataset,
                               EKUT_val_dataset,
                               Eyes_Japan_val_dataset], 

                               optimizer,
                               scheduler,
                               experiment_name='experiment_5',
                               batch_size=32)
    trainer.train(1000)



# expriment 2 batch norm (batchnorm1d, batchnorm2d) + permutaions of frames (shuffle) + scheduler + dropout 0.25 each + LeakyReLU
# experiment 3 same as before but fixes batch of 32 instead of whole sequence (better stability?)
# experiment 4 regresion withou relative coordinates of key points, removed Local Responsenorm, fail
# experiment 5 same as 4 with Mean Per Joint Position Error as loss function