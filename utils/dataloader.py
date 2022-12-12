import random
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.utils import normalize_data, Catergorical2OneHotCoding


class MyDataset(Dataset):
    def __init__(self, filename, is_training=True, args=None):
        super(MyDataset).__init__()
        self.is_training = is_training
        self.args = args

        data = pd.read_csv(filename, header=None)

        self.data_y = data.values[:, -1]
        self.data_y = Catergorical2OneHotCoding( self.data_y.astype(np.int8))

        self.data_x = data.drop(columns=[187])
        self.data_x.columns = range(self.data_x.shape[1])
        self.data_x = self.data_x.values
        std_ = self.data_x.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        self.data_x = (self.data_x - self.data_x.mean(axis=1, keepdims=True)) / std_
        self.data_x = np.expand_dims(self.data_x, axis=-1)


    def __len__(self):
        return self.data_x.shape[0]


    def __getitem__(self, index):
        x =  self.data_x[index]
        y =  self.data_y[index]
        return x, y


if __name__ == "__main__":
    print("Fine")