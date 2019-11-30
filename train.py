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

parser = argparse.ArgumentParser(description='Wordnet gat training script.')
parser.add_argument('--no-cuda', action='store_true',
                    default=True, help='Disables CUDA training.')
parser.add_argument('--resume', action='store_true',
                    default=False, help='Resume training from saved model.')
parser.add_argument('--data', type=str,
                    default='./data/wn_graph', help='Graph data path.')
parser.add_argument('--checkpoint_path', type=str,
                    default='', help='Checkpoint path for resuming.')
parser.add_argument('--model', type=str, default='gat',
                    help='Gnn model.')
parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Hidden channels.')
parser.add_argument('--output', type=int, default=256,
                    help='Output channels.')
parser.add_argument('--n_samples', type=int, default=5,
                    help='Number of negitive sampling.')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# x, x_i, x_j = load_data(args.data)
# edge_index = torch.tensor([x_i, x_j], dtype=torch.long)
# x = torch.tensor(x, dtype=torch.float)
# data = Data(x=x, edge_index=edge_index)

data = data.to(device)

if args.model == 'gat':
    model = WNGat(data.num_node_features,
                  hidden_channels=args.hidden,
                  out_channels=args.output,
                  heads=args.n_heads,
                  dropout=args.dropout,
                  use_checkpoint=False).to(device)
else:
    pass

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

epoch_start = 0
loss_list = []
if args.resume:
    ckp = torch.load(args.checkpoint_path)
    model.load_state_dict(ckp['model_state_dict'])
    optimizer.load_state_dict(ckp['optimizer_state_dict'])
    epoch_start = ckp['epoch']
    loss_list = ckp['loss_list']

train_time = time.time()
print(model)

model.train()
negloss = NEGLoss(data.x, data.edge_index, args.n_samples)
for epoch in range(epoch_start, epoch_start+args.epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = negloss(out, data.edge_index)
    loss_list.append(loss.data)
    loss.backward()
    optimizer.step()

    print(f'Train Epoch: {epoch} \t loss: {loss.data}')
    if epoch > 0 and (epoch + 1) % 2 == 0:
        prefix_sav = f'./model_save/{train_time}'
        if not os.path.exists(prefix_sav):
            os.makedirs(prefix_sav)

        state_dict = {
            'epoch': epoch+1,
            'loss_list': loss_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state_dict, f'{prefix_sav}/{args.model}_{epoch+1}.m5')

torch.save(out.data, f'{prefix_sav}/{args.model}_out.emb')
