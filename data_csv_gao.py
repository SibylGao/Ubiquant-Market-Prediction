import torch
from torch import nn as nn
from torch import tensor as tensor
import numpy as np
import gc
import torch.optim as optim
import pandas as pd

class DataLoader():
    def __init__(self,args):     #  args:[data_path, seq_length, mode...]
        super().__init__()
        self.args = args
        self.df = pd.read_csv(args.data_path)
        self.data = self.df.iloc[1:,:]     
        self.bz = args.seq_lenth

    def __getitem__(self):
        landmarks = self.data.get_chunk(self.bz).as_matrix().astype('float')
        if self.args.mode == "train":
            sample = {}
            sample["time_id"] = landmarks.iloc[:,[1]]    # [row_id,time_id,investment_id,target,f...] 取第二列
            sample["target"] = landmarks.iloc[:,[3]]
            sample["features"] = landmarks.iloc[:,4:]
            return sample
        else:
            ## To do:
            return 0
    def __len__(self):
        return len(self.data)


