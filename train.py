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

from models import WNGat, WNNode2vec
from utils import load_data

parser = argparse.ArgumentParser(description='Wordnet gat training script.')
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='Disables CUDA training.')
parser.add_argument('--resume', action='store_true',
                    default=False, help='Resume training from saved model.')
parser.add_argument('--is_parallel', action='store_true',
                    default=False, help='Resume training from saved model.')
parser.add_argument('--reverse', action='store_true',
                    default=False, help='Revserse the order of sample in randomwalk.')
parser.add_argument('--data', type=str,
                    default='./data/wordnet/edge/synsets', help='Graph data path.')
parser.add_argument('--dictionary', type=str,
                    default='./data/dictionary', help='Graph data path.')
parser.add_argument('--checkpoint_path', type=str,
                    default='', help='Checkpoint path for resuming.')
parser.add_argument('--model', type=str, default='node2vec',
                    help='Gnn model.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for some model such as Node2vec, GraphSage.')
parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Hidden channels.')
parser.add_argument('--output', type=int, default=128,
                    help='Output channels.')
parser.add_argument('--n_samples', type=int, default=5,
                    help='Number of negitive sampling.')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2,
#                     help='Alpha for the leaky_relu.')
parser.add_argument('--walk_length', type=int, default=20,
                    help='Number of head attentions.')
parser.add_argument('--context_size', type=int, default=10,
                    help='Number of head attentions.')
parser.add_argument('--walks_per_node', type=int, default=10,
                    help='Number of head attentions.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

# # debug
# device = 'cuda'
# args.model = 'node2vec'
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]

x, x_i, x_j = load_data(args.data, args.dictionary)
edge_index = torch.tensor([x_i, x_j], dtype=torch.long)
x = torch.tensor(x, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

data = data.to(device)
params = {}

if args.model == 'gat':
    model = WNGat(data.num_node_features,
                  hidden_channels=args.hidden,
                  out_channels=args.output,
                  heads=args.n_heads,
                  dropout=args.dropout,
                  use_checkpoint=False).to(device)
elif args.model == 'node2vec':
    model = WNNode2vec(data,
                       edge_weight=None,
                       embedding_dim=args.output,
                       walk_length=args.walk_length,
                       context_size=args.context_size,
                       walks_per_node=args.walks_per_node,
                       is_parallel=args.is_parallel,
                       reverse=args.reverse).to(device)
    params['batch_size'] = args.batch_size
else:
    pass

model = model.to(device)
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

model.train(epoch_start, args.epochs+epoch_start,
            data, args.n_samples, optimizer, device, **params)
