import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.nn import Node2Vec, SAGEConv
from torch.utils.checkpoint import checkpoint

from layer import GATConv, NEGLoss


def save_model(epoch, model, optimizer, loss_list, prefix_sav) -> None:
    model_name = model.__class__.__name__
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_list[-1]))
    if epoch > 0 and (epoch + 1) % 2 == 0:
        if not os.path.exists(prefix_sav):
            os.makedirs(prefix_sav)

        state_dict = {
            'epoch': epoch+1,
            'loss_list': loss_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads)
        self.conv3 = GATConv(hidden_channels*heads, out_channels, heads)
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

    def train(self,
              epoch_s: int,
              epoch_e: int,
              data: torch_geometric.data.Data,
              n_samples: int,
              optimizer: torch.optim,
              mode: bool = True) -> None:

        train_time = time.time()
        prefix_sav = f'./model_save/WNGat_{train_time}'
        loss_list = []

        super(WNGat, self).train(mode=mode)
        negloss = NEGLoss(data.x, data.edge_index, n_samples)
        for epoch in range(epoch_s, epoch_e):
            optimizer.zero_grad()
            out = self.forward(data.x, data.edge_index)
            loss = negloss(out, data.edge_index)
            loss_list.append(loss.data)
            loss.backward()
            optimizer.step()
            save_model(epoch, self, optimizer, loss_list, prefix_sav)


class WNNode2vec(Node2Vec):
    def train(self,
              epoch_s: int,
              epoch_e: int,
              data: torch_geometric.data.Data,
              n_samples: int,
              optimizer: torch.optim,
              device: str,
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
            for subset in loader:
                optimizer.zero_grad()
                loss = super().loss(data.edge_index, subset.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            rls_loss = total_loss / len(loader)
            loss_list.append(rls_loss)
            save_model(epoch, self, optimizer, loss_list, prefix_sav)

    '''
    def test(self,
             data: torch_geometric.data.Data,
             device: str) -> None:

        super().eval()
        with torch.no_grad():
            z = self.forward(torch.arange(data.num_nodes, device=device))
        acc = super().test(z[data.train_mask], data.y[data.train_mask],
                           z[data.test_mask], data.y[data.test_mask], max_iter=150)
        return acc
    '''


# class WNGraphSage(nn.Module):
#     def train(self,
#               epoch_s: int,
#               epoch_e: int,
#               data: torch_geometric.data.Data,
#               n_samples: int,
#               optimizer: torch.optim,
#               device: str,
#               mode: bool = True,
#               batch_size: int = 256) -> None:
        
#         loader = DataLoader(torch.arange(data.num_nodes),
#                             batch_size=batch_size,
#                             shuffle=True)

#         train_time = time.time()
#         prefix_sav = f'./model_save/WNGat_{train_time}'
#         loss_list = []

#         for epoch in range(epoch_s, epoch_e):
#             super().train()
#             total_loss = 0
#             for subset in loader:
#                 optimizer.zero_grad()
#                 loss = super().loss(data.edge_index, subset.to(device))
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             rls_loss = total_loss / len(loader)
#             loss_list.append(rls_loss)
#             print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, total_loss))
#             save_model(epoch, self, optimizer, loss_list, prefix_sav)