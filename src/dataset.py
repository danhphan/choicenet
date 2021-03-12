import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset

class ChoiceDataset:
    def __init__(self, df, atts_cols, avai_cols, target):
        """
        :param data: the choice data frame
        :param atts_cols: attribute columns of alternatives (like travel time, cost)
        :param avail_cols: availability columns of alternatives
        """
        self.df = df
        self.atts_cols = atts_cols
        self.avai_cols = avai_cols
        # Store numpy array of alternative attributes and availability
        self.atts = self.df[atts_cols].values
        self.avai = self.df[avai_cols].values 
        self.target = self.df[target].values
        # Store column name and its corresponding column index of attribute variables and availability
        self.atts_idxs = {att:idx for att, idx in zip(atts_cols, np.arange(len(atts_cols)))}
        self.avai_idxs = {av:idx  for av, idx  in zip(avai_cols, np.arange(len(avai_cols)))}
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        x = self.atts[index]
        av = self.avai[index]
        y = self.target[index]
        return {"x" : torch.tensor(x, dtype=torch.float32),
                "av" : torch.tensor(av, dtype=torch.float32),
                "y" : torch.tensor(y, dtype=torch.long)}