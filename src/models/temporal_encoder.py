
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DEncoder(nn.Module):
    """1D-CNN encoder for temporal signal snippets."""
    
    def __init__(self, segment_length=300, embedding_dim=64, hidden_channels=None):
        super(CNN1DEncoder, self).__init__()
        
        self.segment_length = segment_length
        self.embedding_dim = embedding_dim
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_channels[0],
            kernel_size=7,
            padding=3,
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[1],
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels[1],
            out_channels=hidden_channels[2],
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.bn3 = nn.BatchNorm1d(hidden_channels[2])
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_channels[2], embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        embeddings = self.fc(x)
        
        return embeddings

