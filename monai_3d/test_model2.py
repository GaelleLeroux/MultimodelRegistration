from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    Invertd,
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
import os
from loaders_csv import LotusDataset
# Configuration et initialisation du modèle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128,256),
    strides=(1, 1, 1, 1),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Charger les poids du modèle
model.load_state_dict(torch.load("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/best_metric_model.pth"))
model.eval()

# Définir les transformations
test_org_transforms = Compose(
    [
        LoadImaged(keys='img'),
        EnsureChannelFirstd(keys='img'),
        Orientationd(keys=['img'], axcodes="RAS"),
        CropForegroundd(keys=['img'], source_key='img'),
    ]
)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys='img',
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/out/", output_postfix="seg", output_ext=".nii.gz", resample=False),
    ]
)

# Charger et transformer l'image
root_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/"
train_files_dataset = LotusDataset("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/train.csv")
val_files_dataset = LotusDataset("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/valid.csv")
test_data_dataset = LotusDataset("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/MRI_Seg/CB_training_csv/test.csv")

train_files = [train_files_dataset[i] for i in range(len(train_files_dataset))]
val_files = [val_files_dataset[i] for i in range(len(val_files_dataset))]
test_data = [test_data_dataset[i] for i in range(len(test_data_dataset))]

# test_data = [{'img': image_path}]
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

# Effectuer l'inférence et sauvegarder la segmentation
with torch.no_grad():
    for test_data in test_org_loader:
        test_inputs = test_data["img"].to(device)
        roi_size = (32, 32, 32)
        sw_batch_size = 1
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

        # Les images segmentées sont sauvegardées automatiquement par le transformateur SaveImaged
