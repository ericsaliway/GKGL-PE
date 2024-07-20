import json
import networkx as nx
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import itertools
import scipy.sparse as sp
##from dgl.nn import GATConv
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.base import DGLError
from typing import Callable, Optional, Tuple, Union
##from dgl.nn import GATConv

# Define the Focal Loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  

class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class LinkPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinkPredictor, self).__init__()
        self.W1 = nn.Linear(input_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(lambda edges: {'score': self.W2(F.relu(self.W1(torch.cat([edges.src['h'], edges.dst['h']], dim=1)))).squeeze(1)})
            return g.edata['score']

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn

class _no_norm_GATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, Tuple[int, int]],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True) -> None:
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        """Set the flag to allow zero in-degree for the graph."""
        self._allow_zero_in_degree = set_value

    def forward(self, graph: DGLGraph, feat: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward computation."""
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting `allow_zero_in_degree` '
                               'to `True` when constructing this module will '
                               'suppress this check and let the users handle '
                               'it by themselves.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if hasattr(self, 'fc_src'):
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'ft': feat_src, 'el': (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)})
            graph.dstdata.update({'er': (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from typing import Union, Tuple, Callable, Optional

class GATConv(nn.Module):
    def __init__(self,
                 in_feats: Union[int, Tuple[int, int]],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True) -> None:
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = dgl.utils.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation
        
        # Add normalization layer
        self.norm = nn.BatchNorm1d(num_heads * out_feats)

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        """Set the flag to allow zero in-degree for the graph."""
        self._allow_zero_in_degree = set_value

    def forward(self, graph: DGLGraph, feat: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward computation."""
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting `allow_zero_in_degree` '
                               'to `True` when constructing this module will '
                               'suppress this check and let the users handle '
                               'it by themselves.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if hasattr(self, 'fc_src'):
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

            graph.srcdata.update({'ft': feat_src, 'el': (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)})
            graph.dstdata.update({'er': (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval

            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)

            # Apply normalization
            rst = rst.view(rst.shape[0], -1)
            rst = self.norm(rst)
            rst = rst.view(rst.shape[0], self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst


class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4, feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        
        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
        
        self.predict = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(g, h).flatten(1)
            h = self.leaky_relu(h)  # Apply LeakyReLU activation
        
        if not self.do_train:
            return h.detach()
        
        logits = self.predict(h)
        return logits


class _norm_GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4, feat_drop=0.0, attn_drop=0.0, do_train=False):
        super(GATModel, self).__init__()
        self.do_train = do_train
        
        assert out_feats % num_heads == 0, "out_feats must be divisible by num_heads"
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # List to hold normalization layers
        
        # First layer
        self.layers.append(GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
        self.norms.append(nn.BatchNorm1d(out_feats))  # Add normalization for the first layer
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_feats, out_feats // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, residual=True, activation=F.leaky_relu, allow_zero_in_degree=True))
            self.norms.append(nn.BatchNorm1d(out_feats))  # Add normalization for hidden layers
        
        self.predict = nn.Linear(out_feats, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, g, features):
        h = features
        for layer, norm in zip(self.layers, self.norms):
            h = self.dropout(h)
            h = layer(g, h).flatten(1)
            h = norm(h)  # Apply normalization
            h = self.leaky_relu(h)  # Apply LeakyReLU activation
        
        if not self.do_train:
            return h.detach()
        
        logits = self.predict(h)
        return logits
