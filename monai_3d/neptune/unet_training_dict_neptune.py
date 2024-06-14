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

import os

import torch

import logger

from transform import TrainTransforms,ValidTransforms
from loaders import LotusDataModule
from model import UNetLightningModule
import argparse


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

def main(args):
    
    ################################################################################################################################################################3

    # define transforms for image and segmentation
    train_transforms = TrainTransforms()
    val_transforms = ValidTransforms()
    
    loader = LotusDataModule(args.train_csv,args.val_csv,args.test_csv,args.batch_size,args.num_workers,args.img_column,args.seg_column,train_transforms,val_transforms)
    loader.setup()
    
    ################################################################################################################################################################3
    # create UNet, DiceLoss and Adam optimizer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        save_last=True,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=args.patience, 
        verbose=True, 
        mode="min"
    )

    callbacks=[early_stop_callback, checkpoint_callback]
    neptune_logger = None

    os.environ['NEPTUNE_API_TOKEN'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ZDQ0NTI4Yi03ZWI3LTRiN2UtODAwMi04MThhYzAwNWJhZDgifQ=='

    neptune_logger = NeptuneLogger(
        project='gaellel/MRICBCT',
        tags=args.neptune_tags,
        api_key=os.environ['NEPTUNE_API_TOKEN']
    )
    

    LOGGER = getattr(logger, args.logger)    
    image_logger = LOGGER(log_steps=args.log_steps)
    callbacks.append(image_logger)
    
    
    ###############################################################################################################################################################3
    model = UNetLightningModule(**vars(args))

    trainer = Trainer(
        logger=neptune_logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),
        reload_dataloaders_every_n_epochs=1
    )
    torch.cuda.empty_cache()

    # trainer.fit(model, datamodule=concat_data)
    try:
        trainer.fit(model, datamodule=loader)
    finally:
        neptune_logger.experiment.wait()


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    parser = argparse.ArgumentParser(description='2D Segmentation MRI training')
    
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--test_csv', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/training/label2/test.csv', help='path csv test')
    input_group.add_argument('--train_csv', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/training/label2/train.csv', help='path train csv')
    input_group.add_argument('--val_csv', type=str, default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/training/label2/valid.csv', help='pat valid csv')
    input_group.add_argument('--img_column', type=str, default='mri', help='name colum for image path')
    input_group.add_argument('--seg_column', type=str, default='seg', help='name colum for segmentation path')
    input_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=25)
    
    input_group.add_argument('--batch_size', type=int, default=2, help='batch_size')
    input_group.add_argument('--num_workers', type=int, default=4, help='number workers')
    input_group.add_argument('--epochs', type=int, default=300, help='number echo')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/output")

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default="Seg,comp1")
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="SegmentationLogger")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)
    log_group.add_argument('--steps', type=int, default=-1, help='Max steps')
    args = parser.parse_args()

    main(args)
