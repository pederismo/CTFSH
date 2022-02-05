from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from datasets.cq500 import CQ500


class CQ500DataModule(pl.LightningDataModule):
    # This dataloader doesn't work for testing because we have a complicated setup for testing
    def __init__(self, data_dir: str = "./", stripped: bool = False, batch_size: int = 2, spatial_size: int = 128):
        super().__init__()
        # all transforms are defined in the dataset class
        self.data_dir = data_dir
        self.stripped = stripped
        self.batch_size = batch_size
        self.spatial_size = spatial_size

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.dataset_train = CQ500(root=self.data_dir, mode="train", stripped=self.stripped, spatial_size=self.spatial_size)
            self.dataset_val = CQ500(root=self.data_dir, mode="val", stripped=self.stripped, spatial_size=self.spatial_size)
        
    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, shuffle=False, batch_size=self.batch_size, pin_memory=True, num_workers=4)