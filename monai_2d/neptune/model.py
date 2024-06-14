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

class UNetLightningModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(1, 1, 1, 1),
            num_res_units=2,
        )

        self.loss_function = monai.losses.DiceLoss(sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        # self.post_trans = Compose([EnsureChannelFirstd(keys=["pred"]), ScaleIntensityd(keys=["pred"])])
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["img"], batch["seg"]
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch["img"], batch["seg"]
        print(f"val_images shape: {val_images.shape}")
        val_outputs = sliding_window_inference(val_images, (48, 48), 4, self.model)
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
        self.dice_metric(y_pred=val_outputs, y=val_labels)

    def on_validation_epoch_end(self):
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_mean_dice", metric, prog_bar=True)