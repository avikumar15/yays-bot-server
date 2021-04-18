import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_sample = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # to later access like dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample
