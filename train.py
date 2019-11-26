import argparse
import glob
import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from models import NEGLoss, WNGat
from utils import load_data

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]

x, x_i, x_j = load_data('./data/wn_graph')
edge_index = torch.tensor([x_i, x_j], dtype=torch.long)
x = torch.tensor(x, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = WNGat(data.num_node_features, 256, 512,
              use_checkpoint=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)

model.train()
negloss = NEGLoss(5)
loss_list = []
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = negloss(out, data.edge_index)
    loss_list.append(loss.data)
    loss.backward()
    optimizer.step()

    print(f'Train Epoch: {epoch} \t loss: {loss.data}')
    if epoch > 0 and epoch % 50 == 0:
        torch.save(out.data, f'./model_save/{epoch}_wn_gat.emb')
        torch.save(model.state_dict(), f'./model_save/{epoch}_wn_gat.m5')
