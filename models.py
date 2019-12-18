import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, SAGEConv
from tqdm import tqdm

from layer import GATConv, NEGLoss
from rw import RandomWalk
from sr_eval.utils import sr_test

EPS = 1e-15


def save_model(epoch, model, optimizer, loss_list, prefix_sav, oup, sr) -> None:
    model_name = model.__class__.__name__
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_list[-1]))
    if epoch > 0 and (epoch + 1) % 10 == 0:
        if not os.path.exists(prefix_sav):
            os.makedirs(prefix_sav)

        state_dict = {
            'epoch': epoch+1,
            'loss_list': loss_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'oup': oup,
            'sr': sr
        }
        torch.save(
            state_dict, f'{prefix_sav}/{epoch+1}.m5')


class WNGat(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.,
            use_checkpoint: bool = False,
            **kwargs) -> None:

        super(WNGat, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels*heads,
                             hidden_channels, heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels*heads,
                             out_channels, heads, dropout=dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, inp, edge_index, size=None):
        if self.use_checkpoint:
            inp = torch.autograd.Variable(inp, requires_grad=True)
            h1 = F.elu(checkpoint(self.conv1, inp, edge_index))
            h2 = F.elu(checkpoint(self.conv2, h1, edge_index))
            oup = F.elu(checkpoint(self.conv3, h2, edge_index))
        else:
            h1 = F.elu(self.conv1(inp, edge_index))
            h2 = F.elu(self.conv2(h1, edge_index))
            oup = F.elu(self.conv3(h2, edge_index))
        return oup

    def emb(self, subset, oup: torch.Tensor = None):
        return oup[subset]

    def train(self,
              epoch_s: int,
              epoch_e: int,
              data: Data,
              n_samples: int,
              optimizer: torch.optim,
              device: torch.device,
              strategy: str = 'max',
              mode: bool = True) -> None:

        train_time = time.time()
        prefix_sav = f'./model_save/WNGat_{train_time}'
        loss_list = []

        

        super().train()
        negloss = NEGLoss(data.x, data.edge_index, n_samples)
        for epoch in range(epoch_s, epoch_e):
            optimizer.zero_grad()
            oup = self.forward(data.x, data.edge_index)
            loss = negloss(oup, data.edge_index)
            loss_list.append(loss.data)
            loss.backward()
            optimizer.step()
            sr_params = {'oup': oup}
            sr_rls = sr_test(device, self.emb, strategy, **sr_params)
            save_model(epoch, self, optimizer, loss_list,
                       prefix_sav, oup, sr=sr_rls)


class WNNode2vec(Node2Vec):
    def __init__(self, data, edge_weight, embedding_dim, walk_length, context_size,
                 walks_per_node=1, p=1, q=1, num_negative_samples=None,
                 is_parallel=False, reverse=False):
        super(WNNode2vec, self).__init__(data.num_nodes, embedding_dim,
                                         walk_length, context_size, walks_per_node, p, q, num_negative_samples)
        self.random_walk = RandomWalk(
            data, edge_weight, is_parallel=is_parallel, reverse=reverse)

    def loss(self, edge_index, edge_weight=None, subset=None):
        r"""Computes the loss for the nodes in :obj:`subset` with negative
        sampling."""
        walk = self.__random_walk__(
            edge_index, edge_weight=edge_weight, subset=subset)
        start, rest = walk[:, 0], walk[:, 1:].contiguous()

        h_start = self.embedding(start).view(
            walk.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            walk.size(0), rest.size(1), self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative sampling loss.
        num_negative_samples = self.num_negative_samples
        if num_negative_samples is None:
            num_negative_samples = rest.size(1)

        neg_sample = torch.randint(self.num_nodes,
                                   (walk.size(0), num_negative_samples),
                                   dtype=torch.long, device=edge_index.device)
        h_neg_rest = self.embedding(neg_sample)

        out = (h_start * h_neg_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def __random_walk__(self, edge_index, edge_weight=None, subset=None):
        if subset is None:
            subset = torch.arange(self.num_nodes, device=edge_index.device)
        subset = subset.repeat(self.walks_per_node)

        data = Data(x=None, edge_index=edge_index)
        rw = self.random_walk.walk(
            subset, walk_length=self.walk_length, p=self.p, q=self.q)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0).to(edge_index.device)

    def train(self,
              epoch_s: int,
              epoch_e: int,
              data: Data,
              n_samples: int,
              optimizer: torch.optim,
              device: torch.device,
              strategy: str = 'max',
              mode: bool = True,
              batch_size: int = 256) -> None:

        loader = DataLoader(torch.arange(data.num_nodes),
                            batch_size=batch_size,
                            shuffle=True)

        train_time = time.time()
        prefix_sav = f'./model_save/WNNode2vec_{train_time}'
        loss_list = []

        for epoch in range(epoch_s, epoch_e):
            super().train()
            total_loss = 0
            print(f'epoch: {epoch}')
            for subset in tqdm(loader):
                optimizer.zero_grad()
                loss = self.loss(data.edge_index, subset=subset.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss = 0
            rls_loss = total_loss / len(loader)
            sr_rls = sr_test(self, device, self.forward, strategy)
            loss_list.append(rls_loss)
            oup = self.forward(torch.arange(
                0, data.num_nodes, device=data.edge_index.device)).data
            save_model(epoch, self, optimizer, loss_list,
                       prefix_sav, oup=oup, sr=sr_rls)