from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

print_config()
print("*"*150)

data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/"
root_dir = "/home/luciacev/Documents/Gaelle/MultimodelRegistration/monai_test/model_output/"

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
num_test_files = 5
num_val_files = 6

assert len(data_dicts) > num_test_files + num_val_files, "Il n'y a pas assez de fichiers pour les ensembles spécifiés."

# Répartition des données entre les ensembles de test, de validation, et d'entraînement
test_data = data_dicts[:num_test_files]
val_files = data_dicts[num_test_files:num_test_files + num_val_files]
train_files = data_dicts[num_test_files + num_val_files:]



print("train_files : ",train_files)
print("val_files : ",val_files)
print("test_data : ",test_data)
print("*"*150)
print("size(train_files) : ",len(train_files))
print("size(val_files) : ",len(val_files))
print("size(test_data) : ",len(test_data))



set_determinism(seed=0)

print("CREATE MODEL, LOSS, OPTIMIZER")
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128,256),
    strides=(1, 1, 1, 1),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Pour les couches linéaires
        num_neurons = module.out_features
        print(f"{name} - Linear layer with {num_neurons} neurons")
    elif isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d):
        # Pour les couches convolutives (2D ou 3D)
        num_neurons = module.out_channels
        print(f"{name} - Convolutional layer with {num_neurons} filters/neurons")



test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=50,
        #     a_max=800,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)

test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
    ]
)


model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
from monai.transforms import LoadImage
loader = LoadImage()
with torch.no_grad():
    for test_data in test_org_loader:
        test_inputs = test_data["image"].to(device)
        roi_size = (32, 32, 32)
        sw_batch_size = 1
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]

#         # uncomment the following lines to visualize the predicted results
        test_output = from_engine(["pred"])(test_data)

        original_image = loader(test_output[0].meta["filename_or_obj"])

        # plt.figure("check", (18, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(original_image[3, :, :], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow()
        # plt.show()
        
        size = original_image.shape
        # print("SIZE TEST OUTOUT :  " ,test_output[0].detach().cpu().shape)
        i = 0
        for m in size :
            i+=1
            for num in range(m):
                plt.figure(figsize=(12, 6))
                if i==1:
                    plt.subplot(1, 2, 1)
                    plt.title("image")
                    plt.imshow(original_image[num, :, :], cmap="gray")
                    plt.subplot(1, 2, 2)
                    plt.title("label")
                    plt.imshow(test_output[0].detach().cpu()[1, num, :, :])
                    plt.savefig(f'./image_test/x/figure{num}_checkdata.png')
                elif i==2:
                    plt.subplot(1, 2, 1)
                    plt.title("image")
                    plt.imshow(original_image[:, num, :], cmap="gray")
                    plt.subplot(1, 2, 2)
                    plt.title("label")
                    plt.imshow(test_output[0].detach().cpu()[1, :, num, :])
                    plt.savefig(f'./image_test/y/figure{num}_checkdata.png')
                elif i==3:
                    plt.subplot(1, 2, 1)
                    plt.title("image")
                    plt.imshow(original_image[:, :, num], cmap="gray")
                    plt.subplot(1, 2, 2)
                    plt.title("label")
                    plt.imshow(test_output[0].detach().cpu()[1, :, :, num])
                    plt.savefig(f'./image_test/z/figure{num}_checkdata.png')
                plt.close()
        break