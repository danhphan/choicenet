import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MNL(nn.Module):
    def __init__(self, atts_idxs, avai_idxs, N=64):
        super(MNL, self).__init__()
        self.atts_idxs = atts_idxs
        self.avai_idxs = avai_idxs
        # Initiate parameters
        self.ASC_TRAIN = nn.Parameter(torch.full((N,), 0.1))
        self.ASC_CAR = nn.Parameter(torch.full((N,), 0.1))
        self.B_TIME = nn.Parameter(torch.full((N,), 0.1))
        self.B_COST = nn.Parameter(torch.full((N,), 0.1))
        
    def forward(self, x, av):
        # Calculate V
        V1 = (self.ASC_TRAIN + 
              self.B_TIME * x[:, self.atts_idxs["TRAIN_TT"]] + 
              self.B_COST * x[:, self.atts_idxs["TRAIN_CO"]])
        V2 = (self.B_TIME * x[:, self.atts_idxs["SM_TT"]] +
              self.B_COST * x[:, self.atts_idxs["SM_CO"]])
        V3 = (self.ASC_CAR + 
              self.B_TIME * x[:, self.atts_idxs["CAR_TT"]] +
              self.B_COST * x[:, self.atts_idxs["CAR_CO"]])
        # Join with availability
        V1 = V1 * av[:, self.avai_idxs["TRAIN_AV"]]
        V2 = V2 * av[:, self.avai_idxs["SM_AV"]]
        V3 = V3 * av[:, self.avai_idxs["CAR_AV"]]
        # Concat into one matrix
        V = torch.cat((V1.unsqueeze(-1), V2.unsqueeze(-1), V3.unsqueeze(-1)),1)
        # Get probality and loglikelihood
        V = V / 100 # A trick so V.exp() will not too big (inf)
        probs = V.exp() / (V.exp().sum(-1, keepdim=True))
        return probs
    
    def string(self):
        return f'ASC_TRAIN={self.ASC_TRAIN}, ASC_CAR={self.ASC_CAR}, B_TIME={self.B_TIME}, B_COST={self.B_COST}'

    
