from argparse import ArgumentParser

import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from datasets.dataset_synthetics import FaceDataset, DataLoaderX


class FaceSynthetics(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.save_hyperparameters()
        backbone = timm.create_model(backbone, num_classes=68*2)
        self.backbone = backbone
        self.loss = nn.L1Loss(reduction='mean')
        self.hard_mining = False

    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        if self.hard_mining:
            loss = torch.abs(y_hat - y) #(B,K)
            loss = torch.mean(loss, dim=1) #(B,)
            B = len(loss)
            S = int(B*0.5)
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:S]
            loss = torch.mean(loss) * 5.0
        else:
            loss = self.loss(y_hat, y) * 5.0
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0002)
        opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in [15, 25, 28] if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]



def cli_main():
    pl.seed_everything(727)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--backbone', default='resnet50d', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--root', default='data/synthetics', type=str)
    parser.add_argument('--num-gpus', default=2, type=int)
    parser.add_argument('--tf32', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------
    # data
    # ------------
    train_set = FaceDataset(root_dir=args.root, is_train=True)
    val_set = FaceDataset(root_dir=args.root, is_train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # ------------
    # model
    # ------------
    model = FaceSynthetics(backbone=args.backbone)
    ckpt_path = 'work_dirs/synthetics'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=10,
            mode='min',
            )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus = args.num_gpus,
        accelerator="ddp",
        benchmark=True,
		logger=TensorBoardLogger(osp.join(ckpt_path, 'logs')),
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=1,
        max_epochs=30,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()

