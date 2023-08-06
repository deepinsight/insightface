import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import timm

class GazeModel(pl.LightningModule):
    def __init__(self, backbone, epoch):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(backbone, num_classes=481*2*3)
        self.epoch = epoch
        #self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.L1Loss(reduction='mean')
        #self.hard_mining = False
        self.hard_mining = False
        self.num_face = 1103
        self.num_eye = 481*2

    def forward(self, x):
        # use forward for inference/predictions
        y = self.backbone(x)
        return y

    def cal_loss(self, y_hat, y, hm=False):
        bs = y.size(0)
        y_hat = y_hat.view( (bs,-1,3) )
        loss = torch.abs(y_hat - y) #(B,K,3)
        loss[:,:,2] *= 0.5
        if hm:
            loss = torch.mean(loss, dim=(1,2)) #(B,)
            loss, _ = torch.topk(loss, k=int(bs*0.25), largest=True)
            #B = len(loss)
            #S = int(B*0.5)
            #loss, _ = torch.sort(loss, descending=True)
            #loss = loss[:S]
        loss = torch.mean(loss) * 20.0
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y, self.hard_mining)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.cal_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0002)
        opt = torch.optim.SGD(self.parameters(), lr = 0.1, momentum=0.9, weight_decay = 0.0005)
        epoch_steps = [int(self.epoch*0.4), int(self.epoch*0.7), int(self.epoch*0.9)]
        print('epoch_steps:', epoch_steps)
        def lr_step_func(epoch):
            return 0.1 ** len([m for m in epoch_steps if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt, lr_lambda=lr_step_func)
        lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'epoch',
                'frequency': 1}
        return [opt], [lr_scheduler]
