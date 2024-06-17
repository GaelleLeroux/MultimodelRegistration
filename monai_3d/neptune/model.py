import os
import torch
import monai
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld, RandRotate90d
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss

class UNetLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128,256),
            strides=(1, 1, 1, 1),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        
        self.loss_function = DiceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Pour les couches linéaires
                num_neurons = module.out_features
                print(f"{name} - Linear layer with {num_neurons} neurons")
            elif isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d):
                # Pour les couches convolutives (2D ou 3D)
                num_neurons = module.out_channels
                print(f"{name} - Convolutional layer with {num_neurons} filters/neurons")
                
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=1)])
        self.post_label = Compose([AsDiscrete()])



    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        # self.optimizer.step()
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch["img"], batch["seg"]
        val_outputs = sliding_window_inference(val_images, (32, 32, 32), 1, self.model)
        val_outputs = [self.post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels = [self.post_label(i) for i in decollate_batch(val_labels)]

        # Convert list of tensors to a single tensor
        val_outputs_tensor = torch.stack(val_outputs)
        val_labels_tensor = torch.stack(val_labels)

        # Ensure the labels have a channel dimension of 1
        if val_labels_tensor.ndim == 4:
            val_labels_tensor = val_labels_tensor.unsqueeze(1)

        loss = self.loss_function(val_outputs_tensor, val_labels_tensor)
        self.dice_metric(y_pred=val_outputs_tensor, y=val_labels_tensor)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_mean_dice", metric, prog_bar=True)