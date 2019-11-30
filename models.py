import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layer import GATConv, NEGLoss


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
            **kwargs
    ) -> None:
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
