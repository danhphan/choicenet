import torch
# import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset, model, engine

# Read data
df_train = pd.read_csv("../data/swissmetro_train.csv")
df_train = df_train[:-60]
df_valid = pd.read_csv("../data/swissmetro_valid.csv")
df_valid = df_valid[:-4]
df_train.shape, df_valid.shape

# Prepare datasets
atts_cols = ['TRAIN_TT', 'TRAIN_CO','SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO']
avai_cols = ['TRAIN_AV', 'CAR_AV', 'SM_AV']

target = "CHOICE"
ds_train = dataset.ChoiceDataset(df_train, atts_cols, avai_cols, target)
ds_valid = dataset.ChoiceDataset(df_valid, atts_cols, avai_cols, target)

atts_idxs = ds_train.atts_idxs
avai_idxs = ds_train.avai_idxs

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=4)
dl_valid = DataLoader(ds_valid, batch_size=64, shuffle=True, num_workers=4)

# Training and evaluation
epochs = 10
lr = 1e-9

model = model.MNL(atts_idxs, avai_idxs)
optimizer = optim.Adam(model.parameters(),lr=lr)

device = "cuda"

for epoch in range(epochs):
    engine.train(dl_train, model, optimizer, device)
    preds, targets = engine.evaluate(dl_valid, model, device)
