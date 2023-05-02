import pytorch_lightning as pl
import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from schedulers.transformer_scheduler import get_scheduler, SchedulerLateTotalstepsSetter
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


class SimpleDataset(Dataset):
    def __init__(self):
        X = np.arange(10000)
        y = X * 2
        X = [[_] for _ in X]
        y = [[_] for _ in y]
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]}


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.criterion = MSELoss()

    def forward(self, inputs_id, labels=None):
        outputs = self.fc(inputs_id)
        loss = 0
        if labels is not None:
            loss = self.criterion(outputs, labels)
        return loss, outputs

    def train_dataloader(self):
        dataset = SimpleDataset()
        return DataLoader(dataset, batch_size=1000)

    def training_step(self, batch, batch_idx):
        input_ids = batch["X"]
        labels = batch["y"]
        loss, outputs = self(input_ids, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer
    
    def configure_optimizers(self):
        scheduler_config = {
            'max_lr': 0.001,
            'warmup_steps': 80,
            'constant_from_epoch': 30,
            'constant_lr': 0.001
        }
        
        optimizer = AdamW(self.parameters(), lr=scheduler_config['max_lr']) # The lr will be fixed by the scheduler
        scheduler = get_scheduler(optimizer, 'cos', **scheduler_config)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


if __name__ == '__main__':
    model = MyModel()
    logger = TensorBoardLogger(save_dir='.', name='logs')
    
    callbacks = [LearningRateMonitor(logging_interval='step'), SchedulerLateTotalstepsSetter(length_from='dataloader')]
    
    trainer = pl.Trainer(max_epochs=40, gpus=1, logger=logger, callbacks=callbacks)
    trainer.fit(model)