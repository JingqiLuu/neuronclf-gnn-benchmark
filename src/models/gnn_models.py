#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN model definitions.

Includes GCN, GraphSAGE, GAT, and GraphTransformer models.
Supports temporal encoder for dynamic signal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, NNConv

from .temporal_encoder import CNN1DEncoder


def encode_temporal_snippets(temporal_encoder, x):
    """Encode temporal snippets using temporal encoder."""
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        N, S, L = x.shape
        snippets_flat = x.view(N * S, L)
        embeddings_flat = temporal_encoder(snippets_flat)
        embeddings = embeddings_flat.view(N, S, -1)
        node_features = embeddings.mean(dim=1)
        return node_features
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")


class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, use_edge_attr=False,
                 batch_norm=True, residual_connection=False, activation='relu', aggr='mean', 
                 bias=True, normalize=True, use_temporal_encoder=False, segment_length=300):
        super(GCN, self).__init__()
        
        self.use_edge_attr = use_edge_attr
        self.use_temporal_encoder = use_temporal_encoder
        
        # 时序编码器（如果启用）
        if use_temporal_encoder:
            self.temporal_encoder = CNN1DEncoder(
                segment_length=segment_length,
                embedding_dim=input_dim
            )
        else:
            self.temporal_encoder = None
        
        self.convs = nn.ModuleList()
        
        if use_edge_attr:
            # 使用NNConv支持边权重
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),  # 边权重维度为1
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim * hidden_dim)
            )
            self.convs.append(NNConv(input_dim, hidden_dim, edge_nn, aggr='mean'))
            
            for _ in range(num_layers - 2):
                edge_nn = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.convs.append(NNConv(hidden_dim, output_dim, edge_nn, aggr='mean'))
        else:
            # 原始GCN实现
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr=None):
        if self.temporal_encoder is not None:
            x = encode_temporal_snippets(self.temporal_encoder, x)
        
        for i, conv in enumerate(self.convs[:-1]):
            if self.use_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_edge_attr and edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, use_edge_attr=False,
                 batch_norm=True, residual_connection=False, activation='relu', aggr='mean', 
                 bias=True, normalize=True, use_temporal_encoder=False, segment_length=300):
        super(GraphSAGE, self).__init__()
        
        self.use_edge_attr = use_edge_attr
        self.batch_norm = batch_norm
        self.residual_connection = residual_connection
        self.activation = activation
        self.aggr = aggr
        self.use_temporal_encoder = use_temporal_encoder
        
        if use_temporal_encoder:
            self.temporal_encoder = CNN1DEncoder(
                segment_length=segment_length,
                embedding_dim=input_dim
            )
        else:
            self.temporal_encoder = None
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        if use_edge_attr:
            # 使用NNConv支持边权重
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim * hidden_dim)
            )
            self.convs.append(NNConv(input_dim, hidden_dim, edge_nn, aggr='mean'))
            
            for _ in range(num_layers - 2):
                edge_nn = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.convs.append(NNConv(hidden_dim, output_dim, edge_nn, aggr='mean'))
        else:
            # 原始GraphSAGE实现
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
        if batch_norm:
            for i in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else output_dim))
        
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif activation == 'elu':
            self.activation_fn = F.elu
        elif activation == 'gelu':
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu
        
    def forward(self, x, edge_index, edge_attr=None):
        # 如果使用时序编码器，先对原始片段进行编码
        if self.temporal_encoder is not None:
            x = encode_temporal_snippets(self.temporal_encoder, x)
        
        for i, conv in enumerate(self.convs[:-1]):
            # 保存残差连接的输入
            if self.residual_connection and i > 0:
                residual = x
            
            if self.use_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            if self.batch_norm and self.batch_norms is not None and self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            x = self.activation_fn(x)
            
            # 残差连接
            if self.residual_connection and i > 0 and x.size() == residual.size():
                x = x + residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        if self.use_edge_attr and edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)
            
        # 最后一层的批归一化
        if self.batch_norm and self.batch_norms is not None and self.batch_norms[-1] is not None:
            x = self.batch_norms[-1](x)
            
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=8, dropout=0.5, use_edge_attr=False,
                 batch_norm=True, residual_connection=False, activation='relu', aggr='mean', 
                 bias=True, normalize=True, concat=True, negative_slope=0.2, add_self_loops=True, edge_dim=1, use_temporal_encoder=False, segment_length=300):
        super(GAT, self).__init__()
        
        self.use_edge_attr = use_edge_attr
        self.use_temporal_encoder = use_temporal_encoder
        
        if use_temporal_encoder:
            self.temporal_encoder = CNN1DEncoder(
                segment_length=segment_length,
                embedding_dim=input_dim
            )
        else:
            self.temporal_encoder = None
        
        self.batch_norm = batch_norm
        self.residual_connection = residual_connection
        self.activation = activation
        self.aggr = aggr
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        if use_edge_attr:
            # 使用NNConv支持边权重
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim * hidden_dim)
            )
            self.convs.append(NNConv(input_dim, hidden_dim, edge_nn, aggr='mean'))
            
            for _ in range(num_layers - 2):
                edge_nn = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.convs.append(NNConv(hidden_dim, output_dim, edge_nn, aggr='mean'))
        else:
            # 原始GAT实现
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
        
        self.dropout = dropout
        
        if batch_norm:
            for i in range(num_layers):
                if i < num_layers - 1:
                    bn_dim = hidden_dim * heads if not use_edge_attr else hidden_dim
                else:
                    bn_dim = output_dim
                if bn_dim > 1:
                    self.batch_norms.append(nn.BatchNorm1d(bn_dim))
                else:
                    self.batch_norms.append(None)
        
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif activation == 'elu':
            self.activation_fn = F.elu
        elif activation == 'gelu':
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu
        
    def forward(self, x, edge_index, edge_attr=None):
        # 如果使用时序编码器，先对原始片段进行编码
        if self.temporal_encoder is not None:
            x = encode_temporal_snippets(self.temporal_encoder, x)
        
        for i, conv in enumerate(self.convs[:-1]):
            # 保存残差连接的输入
            if self.residual_connection and i > 0:
                residual = x
            
            if self.use_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            if self.batch_norm and self.batch_norms is not None and self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            x = self.activation_fn(x)
            
            # 残差连接
            if self.residual_connection and i > 0 and x.size() == residual.size():
                x = x + residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        if self.use_edge_attr and edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)
            
        # 最后一层的批归一化
        if self.batch_norm and self.batch_norms is not None and self.batch_norms[-1] is not None:
            x = self.batch_norms[-1](x)
            
        return x


class GraphTransformer(nn.Module):
    """Graph Transformer Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.2, use_edge_attr=False,
                 batch_norm=True, residual_connection=False, activation='relu', aggr='mean', 
                 bias=True, normalize=True, concat=True, beta=0.0, edge_dim=1, root_weight=True, use_temporal_encoder=False, segment_length=300):
        super(GraphTransformer, self).__init__()
        
        self.use_edge_attr = use_edge_attr
        self.use_temporal_encoder = use_temporal_encoder
        
        # 时序编码器（如果启用）
        if use_temporal_encoder:
            self.temporal_encoder = CNN1DEncoder(
                segment_length=segment_length,
                embedding_dim=input_dim
            )
        else:
            self.temporal_encoder = None
        
        self.batch_norm = batch_norm
        self.residual_connection = residual_connection
        self.activation = activation
        self.aggr = aggr
        self.concat = concat
        self.beta = beta
        self.edge_dim = edge_dim
        self.root_weight = root_weight
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        if use_edge_attr:
            # 使用NNConv支持边权重
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim * hidden_dim)
            )
            self.convs.append(NNConv(input_dim, hidden_dim, edge_nn, aggr='mean'))
            
            for _ in range(num_layers - 2):
                edge_nn = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
                )
                self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            
            edge_nn = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * output_dim)
            )
            self.convs.append(NNConv(hidden_dim, output_dim, edge_nn, aggr='mean'))
        else:
            # 原始GraphTransformer实现
            self.convs.append(TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            
            for _ in range(num_layers - 2):
                self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            
            self.convs.append(TransformerConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
        
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        if batch_norm:
            for i in range(num_layers):
                if i < num_layers - 1:
                    bn_dim = hidden_dim * heads if not use_edge_attr else hidden_dim
                else:
                    bn_dim = output_dim
                if bn_dim > 1:
                    self.batch_norms.append(nn.BatchNorm1d(bn_dim))
                else:
                    self.batch_norms.append(None)
        
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu
        elif activation == 'elu':
            self.activation_fn = F.elu
        elif activation == 'gelu':
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu
        
    def forward(self, x, edge_index, edge_attr=None):
        # 如果使用时序编码器，先对原始片段进行编码
        if self.temporal_encoder is not None:
            x = encode_temporal_snippets(self.temporal_encoder, x)
        
        for i, conv in enumerate(self.convs[:-1]):
            # 保存残差连接的输入
            if self.residual_connection and i > 0:
                residual = x
            
            if self.use_edge_attr and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            if self.batch_norm and self.batch_norms is not None and self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            
            # 激活函数
            x = self.activation_fn(x)
            
            # 残差连接
            if self.residual_connection and i > 0 and x.size() == residual.size():
                x = x + residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        if self.use_edge_attr and edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)
            
        # 最后一层的批归一化
        if self.batch_norm and self.batch_norms is not None and self.batch_norms[-1] is not None:
            x = self.batch_norms[-1](x)
            
        return x
