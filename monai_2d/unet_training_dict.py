# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image

import argparse
import glob 
import matplotlib.pyplot as plt

def main(args):
    data_dir = args.data
    model_dir = args.model

    pattern_images = os.path.join(data_dir, "ImagesTr","**", "*.png")
    pattern_labels = os.path.join(data_dir, "LabelsTr","**", "*.png")
    train_images = sorted(glob.glob(pattern_images,recursive=True))
    train_labels = sorted(glob.glob(pattern_labels,recursive=True))
    

    data_dicts = [{"img": image_name, "seg": label_name} for image_name, label_name in zip(train_images, train_labels)]
    num_test_files = 0
    num_val_files = 6

    assert len(data_dicts) > num_test_files + num_val_files, "Il n'y a pas assez de fichiers pour les ensembles spécifiés."

    # Répartition des données entre les ensembles de test, de validation, et d'entraînement
    test_data = data_dicts[:num_test_files]
    val_files = data_dicts[num_test_files:num_test_files + num_val_files]
    train_files = data_dicts[num_test_files + num_val_files:]

    print("train_files : ",train_files)
    print("*"*150)
    print("val_files : ",val_files)
    

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[48, 48], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys=["img", "seg"]),
        ]
    )

    # define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["seg"].shape)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(args.echo):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.echo}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (48, 48)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.model,"best_metric_model_segmentation2d_dict.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    
    plt.figure(figsize=(12, 6))
    
    # Graphique pour la loss moyenne par époque
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, label="Train Loss", color='blue')
    plt.legend()
    
    # Graphique pour le mean dice score
    plt.subplot(1, 2, 2)
    plt.title("Validation Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice Score")
    plt.plot([val_interval * (i + 1) for i in range(len(metric_values))], metric_values, label="Val Mean Dice", color='red')
    plt.legend()
    
    # Sauvegarde de la figure
    plt.savefig(os.path.join(model_dir, os.path.join(args.model,'training_performance.png')))
    plt.close()


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--data', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/', help='Input folder')
    parser.add_argument('--model', type=str, default='/home/luciacev/Documents/Gaelle/MultimodelRegistration/monai_2d/output_model/', help='Output directory tosave the png')
    parser.add_argument('--echo', type=int, default=600, help='number echo')
    args = parser.parse_args()

    main(args)