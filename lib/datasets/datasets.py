import torch
from torch.utils.data import Dataset
from . import dataset_utils
import numpy as np
import torchvision.datasets
import torchvision.transforms
import os


@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


@dataset_utils.register_dataset
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S
        L = cfg.data.shape[0]
        np_data = np.load(cfg.data.path) # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]