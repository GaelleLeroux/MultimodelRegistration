
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from monai.data import list_data_collate, DataLoader
import pandas as pd

import pandas as pd

import monai

import pytorch_lightning as pl
import torch
from monai.transforms import (  
    LoadImaged
)   


class LotusDataset(Dataset):
    def __init__(self, path_csv="./", img_column="mri", seg_column="seg", transform=None):
        self.img_column = img_column
        self.seg_column = seg_column
        self.transform = transform
        self.df = pd.read_csv(path_csv)  # Charger le DataFrame ici
        self.load = LoadImaged(keys=["img", "seg"])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_dict = {
            "img": row[self.img_column],
            "seg": row[self.seg_column]
        }

        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
class LotusDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, batch_size=2, num_workers=2, img_column="mri", seg_column="seg", train_transform=None, valid_transform=None, test_transform=None):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column    
        self.seg_column = seg_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform      

    def setup(self, stage=None):
        self.train_ds = monai.data.CacheDataset(data=LotusDataset(self.train_csv, img_column=self.img_column, seg_column=self.seg_column), transform=self.train_transform, cache_rate=1.0, num_workers=self.num_workers)
        self.val_ds = monai.data.CacheDataset(data=LotusDataset(self.val_csv, img_column=self.img_column, seg_column=self.seg_column), transform=self.valid_transform, cache_rate=1.0, num_workers=self.num_workers)
        self.test_ds = monai.data.CacheDataset(data=LotusDataset(self.test_csv, img_column=self.img_column, seg_column=self.seg_column), transform=self.test_transform, cache_rate=1.0, num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds