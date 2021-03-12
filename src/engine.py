import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
loss_func = F.cross_entropy


def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()


def nll(probs, targets): 
    log_likelihood = probs[:, targets -1 ].log().sum()       
    return - log_likelihood

def train(data_loader, model, optimizer, device):
    model.train()
    total = 0
    sum_loss = 0
    correct = 0
    for data in data_loader:
        x = data['x']
        av = data['av']
        batch_sz = x.shape[0]
#         x = x.to(device, dtype=torch.float32)
#         av = av.to(device, dtype=torch.float32)
        targets = data['y'] 
        
        outputs = model(x, av)   
        loss = nll(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += batch_sz*(loss.item())
        total += batch_sz
        preds = torch.argmax(outputs, 1)
        correct += (preds == (targets - 1)).float().sum().item()
        test =  (torch.argmax(outputs, dim=1)==(targets -1)).float()

        
    print("Train loss: %.5f , NLL: %.5f , Accuracy: %.5f" % (sum_loss/total, sum_loss/total/total, correct/total))
        
def evaluate(data_loader, model, device):
    model.eval()
    final_targets = []
    final_outputs = []
    total = 0
    sum_loss = 0
    correct = 0
    with torch.no_grad():        
        for data in data_loader:            
            x = data['x']
            av = data['av']
            batch_sz = x.shape[0]
#             x = x.to(device, dtype=torch.float32)
#             av = av.to(device, dtype=torch.float32)
            targets = data['y']
            outputs = model(x, av)

            loss = nll(outputs, targets)
            sum_loss += loss
            total += batch_sz
            preds = torch.argmax(outputs, 1)
            correct += (preds == (targets - 1)).float().sum().item()

            final_outputs.extend(outputs)
            final_targets.extend(targets)

    # print("Valid loss: %.5f and accuracy: %.5f" % (sum_loss/total, correct/total))
    print("Valid loss: %.5f , NLL: %.5f , Accuracy: %.5f" % (sum_loss/total, sum_loss/total/total, correct/total))
    return final_outputs, final_targets