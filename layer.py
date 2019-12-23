from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (add_self_loops, degree, remove_self_loops,
                                   softmax, structured_negative_sampling)


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        t = self.propagate(edge_index, size=size, x=x)
        return t

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class NEGLoss(nn.Module):
    def __init__(self, x: torch.FloatTensor, edge_index: torch.LongTensor, num_negative_samples: int = 5):
        super(NEGLoss, self).__init__()
        self.num_negative_samples = num_negative_samples
        self.bce_loss_with_logits = nn.BCEWithLogitsLoss()
        self.neg_edge_index = self.sample(edge_index, x.size(0))

    def sample(self, edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
        neg_edge_index = []
        for i in range(self.num_negative_samples):
            tmp = structured_negative_sampling(edge_index, num_nodes)
            neg_edge_index.append(tmp[-1].tolist())
        return torch.LongTensor(neg_edge_index)

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor) -> float:
        neg_edge_index = self.neg_edge_index
        v_i = x[edge_index[0]]
        nei_v_i = x[edge_index[1]]
        pos = torch.sum(v_i.mul(nei_v_i), dim=1, dtype=torch.float)

        tv_i = v_i.repeat(1, self.num_negative_samples).reshape(-1, v_i.size(1))
        neg_v_i = x[torch.transpose(
            neg_edge_index, 1, 0)].reshape(-1, v_i.size(1))
        neg = torch.sum(torch.mul(tv_i, neg_v_i), dim=1)

        loss = self.bce_loss_with_logits(torch.cat((pos, neg)), torch.cat(
            (torch.ones_like(pos), torch.zeros_like(neg))))
        return loss
