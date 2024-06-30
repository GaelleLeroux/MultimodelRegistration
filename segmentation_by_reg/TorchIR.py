import matplotlib.pyplot as plt
from pathlib import Path
DEST_DIR = Path('./output')


import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchir.utils import IRDataSet

class MNISTSubSet(MNIST):
    '''
    A Dataset class that selects a single type of MNIST digit.
    '''
    def __init__(self, label, rng=np.random.default_rng(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(label >= 0 and label <= 9)
        idcs = torch.where(self.targets == label)
        
        self.data = self.data[idcs]
        self.targets = self.targets[idcs]

        self.transform = transform
        self.rng = rng
        
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0] # only return image


rng = np.random.default_rng(808)
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5), (0.5)),
                             ])
ds_train_subset = MNISTSubSet(label=9, rng=rng, root='../datasets/',  transform=transform, download=True, train=True)


# val_set_size = 20
# train_set_size = len(ds_train_subset) - val_set_size
# ds_train_subset, ds_validation_subset = torch.utils.data.random_split(ds_train_subset, [train_set_size, val_set_size], 
#                                                         generator=torch.Generator().manual_seed(808))
# ds_test_subset = MNISTSubSet(label=9, rng=rng, root='../datasets/',  transform=transform, download=True, train=False)
# print(f'Training subset size: {len(ds_train_subset)}')
# print(f'Validation subset size: {len(ds_validation_subset)}')
# print(f'Test subset size: {len(ds_test_subset)}')

# ds_train = IRDataSet(ds_train_subset)
# ds_validation = IRDataSet(ds_validation_subset)
# print(f'Training IR set size: {len(ds_train)}')
# print(f'Validation IR set size: {len(ds_validation)}')

# batch_size = 32
# training_batches = 100
# validation_batches = 10

# train_sampler = torch.utils.data.RandomSampler(ds_train, replacement=True, 
#                                                num_samples=training_batches*batch_size, 
#                                                generator=torch.Generator().manual_seed(808))
# train_loader = torch.utils.data.DataLoader(ds_train, batch_size, sampler=train_sampler)
# val_loader = torch.utils.data.DataLoader(ds_validation, batch_size)

