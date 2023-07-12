import torch
from torch import nn

# net 1 no batchnorm
# net 2 added batchnorm


class Network(nn.Module):
    def __init__(self, number_of_joints):
        super(Network, self).__init__()
        self.joints = number_of_joints

        self.features = nn.Sequential(
            nn.Conv2d(3, 3, 2, 2, 0), # in: 224, out: 112
            #nn.BatchNorm2d(3),
            nn.ReLU(),
            
            nn.Conv2d(3, 32, 9, 1, 0), # in: 112, out: 104
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # in: 104, out: 52

            nn.Conv2d(32, 64, 5, 1, 0), # in: 52, out: 48
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # in: 48, out: 24

            nn.Conv2d(64, 64, 5, 1, 0), # in: 24, out: 20
            #nn.BatchNorm2d(64),
            torch.nn.LocalResponseNorm(5),
            nn.MaxPool2d(2, 2, 0), # in: 20, out: 10
        )
        

        self.regression = nn.Sequential(
            # similarly as on the last exercise
            nn.Linear(64 * 10 * 10, 1024),
            #nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, number_of_joints * 3),
            #nn.BatchNorm1d(number_of_joints * 3),
            nn.Tanh()
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.regression.apply(init_weights)
        self.features.apply(init_weights)

    def forward(self, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        if type(sample) is dict:
            x = sample['sequences']
        else:
            x = sample
        if x.dim() == 5:
            x = torch.squeeze(x, 0)
        s, c, h, w, = x.shape
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression(x)
        return torch.reshape(x, (s, 3, self.joints))
    

class NetworkBatchNorm(nn.Module):
    def __init__(self, number_of_joints):
        super(NetworkBatchNorm, self).__init__()
        self.joints = number_of_joints

        self.features = nn.Sequential(
            nn.Conv2d(3, 3, 2, 2, 0), # in: 224, out: 112
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            
            nn.Conv2d(3, 32, 9, 1, 0), # in: 112, out: 104
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0), # in: 104, out: 52

            nn.Conv2d(32, 64, 5, 1, 0), # in: 52, out: 48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0), # in: 48, out: 24

            nn.Conv2d(64, 64, 5, 1, 0), # in: 24, out: 20
            nn.BatchNorm2d(64),
            torch.nn.LocalResponseNorm(5),
            nn.MaxPool2d(2, 2, 0), # in: 20, out: 10
        )
        

        self.regression = nn.Sequential(
            # similarly as on the last exercise
            nn.Linear(64 * 10 * 10, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(2048, number_of_joints * 3),
            nn.Tanh()
        )

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.kaiming_uniform_(m.weight)

        self.regression.apply(init_weights)
        self.features.apply(init_weights)

    def forward(self, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        if type(sample) is dict:
            x = sample['sequences']
        else:
            x = sample
        if x.dim() == 5:
            x = torch.squeeze(x, 0)
        s, c, h, w, = x.shape
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression(x)
        return torch.reshape(x, (s, 3, self.joints))