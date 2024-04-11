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
num_test_files = 0
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


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=50,
        #     a_max=800,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(0.46, 0.46, 2.46), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(7, 16, 16),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     # a_min=50,
        #     # a_max=350,
        #     a_min=50,
        #     a_max=800,
        #     b_min=0.0,
        #     b_max=1.0,
        #     clip=True,
        # ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(keys=["image", "label"], pixdim=(0.46, 0.46, 2.46), mode=("bilinear", "nearest")),
    ]
)

print("-"*150)
print("DATA CHECK")

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
print(check_data.keys())
print(check_data['image'].meta.keys()) # Si vous utilisez une version de MONAI où les métadonnées sont directement attachées à l'objet Tensor

image_path = check_data["image"].meta['filename_or_obj']
label_path = check_data["label"].meta['filename_or_obj']

print(f"Image path: {image_path}")
print(f"Label path: {label_path}")
# plot the slice [:, :, 80]
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, 8, :], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:, 8, :])

# plt.savefig('figure1_checkdata.png')
# plt.clf()

size = image.shape
print("size : ",size)
# i = 0
# for m in size :
#     i+=1
#     for num in range(m):
#         plt.figure(figsize=(12, 6))
#         if i==1:
#             plt.subplot(1, 2, 1)
#             plt.title("image")
#             plt.imshow(image[num, :, :], cmap="gray")
#             plt.subplot(1, 2, 2)
#             plt.title("label")
#             plt.imshow(label[num, :, :])
#             plt.savefig(f'./image_mine_50_800/x/demo_figure{num}_checkdata.png')
#         elif i==2:
#             plt.subplot(1, 2, 1)
#             plt.title("image")
#             plt.imshow(image[:, num, :], cmap="gray")
#             plt.subplot(1, 2, 2)
#             plt.title("label")
#             plt.imshow(label[:, num, :])
#             plt.savefig(f'./image_mine_50_800/y/demo_figure{num}_checkdata.png')
#         elif i==3:
#             plt.subplot(1, 2, 1)
#             plt.title("image")
#             plt.imshow(image[:, :, num], cmap="gray")
#             plt.subplot(1, 2, 2)
#             plt.title("label")
#             plt.imshow(label[:, :, num])
#             plt.savefig(f'./image_mine_50_800/z/demo_figure{num}_checkdata.png')
#         plt.close()

# reponse = input("Tapez quelque chose et appuyez sur Entrée : ")
print("-"*150)
print("DATA CHECK END")

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
# train_ds = Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)



####################################################################################################################################################
# CREATE MODEL, LOSS, OPTIMIZER
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
print("-"*150)
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


###################################################################################################################################################
# CLASSIC TRAINING
print("-"*150)
print("CLASSIC TRAINING")
max_epochs = 600
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    torch.cuda.empty_cache() 
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        # inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        print(f"Shape of batch inputs: {inputs.shape}, Shape of batch labels: {labels.shape}")
        # input2 = inputs[0,0,:,:,:]
        outputs = model(inputs)
        print(f"Shape of batch outputs: {outputs.shape}")
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        torch.cuda.empty_cache() 
        model.eval()
        with torch.no_grad():
            num = 0
            for val_data in val_loader:
                num+=1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (32, 32, 32)
                sw_batch_size = 1
                print(f"{num}1")
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                print(f"{num}2")
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                print(f"{num}3")
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                print(f"{num}4")
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                print(f"{num}5")

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
                torch.cuda.empty_cache() 
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            
            
print("-"*150)
print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")


plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.savefig('figure2_loss_metric.png')
plt.clf()
# plt.show()

####################################################################################################################################################
# Check best model output with the input image and label
print("-"*150)
print("Check best model output with the input image and label")


model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
num_fig = 2
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        num_fig+=1
        roi_size = (32, 32, 32)
        sw_batch_size = 1
        val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :, 5], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 5])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 5])
        # plt.show()
        plt.savefig(f'figure{num_fig}_bestmodel_with_inputimagelabel.png')
        plt.clf()
        if i == 2:
            break
        
###################################################################################################################################################
# Evaluation on original image spacings
print("-"*150)
print("Evaluation on original image spacings")
      
val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=50,
            a_max=800,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)

val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
    ]
)

model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        roi_size = (32, 32, 32)
        sw_batch_size = 1
        val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)

    # aggregate the final mean dice result
    metric_org = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

print("Metric on original image spacing: ", metric_org)

####################################################################################################################################################
# Evaluation on original image spacings




# test_org_transforms = Compose(
#     [
#         LoadImaged(keys="image"),
#         EnsureChannelFirstd(keys="image"),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=50,
#             a_max=800,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),
#         CropForegroundd(keys=["image"], source_key="image"),
#     ]
# )

# test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

# test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

# post_transforms = Compose(
#     [
#         Invertd(
#             keys="pred",
#             transform=test_org_transforms,
#             orig_keys="image",
#             meta_keys="pred_meta_dict",
#             orig_meta_keys="image_meta_dict",
#             meta_key_postfix="meta_dict",
#             nearest_interp=False,
#             to_tensor=True,
#         ),
#         AsDiscreted(keys="pred", argmax=True, to_onehot=2),
#         SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
#     ]
# )


# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# from monai.transforms import LoadImage
# loader = LoadImage()
# with torch.no_grad():
#     for test_data in test_org_loader:
#         test_inputs = test_data["image"].to(device)
#         roi_size = (32, 32, 32)
#         sw_batch_size = 1
#         test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

#         test_data = [post_transforms(i) for i in decollate_batch(test_data)]

# #         # uncomment the following lines to visualize the predicted results
#         test_output = from_engine(["pred"])(test_data)

#         original_image = loader(test_output[0].meta["filename_or_obj"])

#         # plt.figure("check", (18, 6))
#         # plt.subplot(1, 2, 1)
#         # plt.imshow(original_image[3, :, :], cmap="gray")
#         # plt.subplot(1, 2, 2)
#         # plt.imshow()
#         # plt.show()
        
#         size = original_image.shape
#         # print("SIZE TEST OUTOUT :  " ,test_output[0].detach().cpu().shape)
#         i = 0
#         for m in size :
#             i+=1
#             for num in range(m):
#                 plt.figure(figsize=(12, 6))
#                 if i==1:
#                     plt.subplot(1, 2, 1)
#                     plt.title("image")
#                     plt.imshow(original_image[num, :, :], cmap="gray")
#                     plt.subplot(1, 2, 2)
#                     plt.title("label")
#                     plt.imshow(test_output[0].detach().cpu()[1, num, :, :])
#                     plt.savefig(f'./image_test/x/demo_figure{num}_checkdata.png')
#                 elif i==2:
#                     plt.subplot(1, 2, 1)
#                     plt.title("image")
#                     plt.imshow(original_image[:, num, :], cmap="gray")
#                     plt.subplot(1, 2, 2)
#                     plt.title("label")
#                     plt.imshow(test_output[0].detach().cpu()[1, :, num, :])
#                     plt.savefig(f'./image_test/y/demo_figure{num}_checkdata.png')
#                 elif i==3:
#                     plt.subplot(1, 2, 1)
#                     plt.title("image")
#                     plt.imshow(original_image[:, :, num], cmap="gray")
#                     plt.subplot(1, 2, 2)
#                     plt.title("label")
#                     plt.imshow(test_output[0].detach().cpu()[1, :, :, num])
#                     plt.savefig(f'./image_test/z/demo_figure{num}_checkdata.png')
#                 plt.close()