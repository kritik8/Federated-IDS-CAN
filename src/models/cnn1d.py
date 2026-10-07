"""
Lightweight 1D-CNN Model for Automotive CAN Bus Intrusion Detection
Optimized for edge in-vehicle execution and Federated Learning aggregation.
Uses GroupNorm to prevent Non-IID client drift in decentralized optimization.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_norm(num_channels: int, max_groups: int = 4) -> nn.GroupNorm:
    """
    Construct nn.GroupNorm with num_groups dividing num_channels.
    
    In Federated Learning with Non-IID client data, BatchNorm causes catastrophic
    divergence during aggregation because clients develop disparate local running
    mean and variance statistics. GroupNorm computes statistics independently per sample
    across channel groups, ensuring stability across distributed vehicular clients.
    
    Args:
        num_channels: Total number of feature channels in layer.
        max_groups: Upper bound on group count (default 4).
        
    Returns:
        Configured nn.GroupNorm layer.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class CAN1DCNN(nn.Module):
    """
    Lightweight 1D Convolutional Neural Network for CAN Bus Intrusion Detection.
    
    Architecture:
      Conv1D(11 -> 32, k=3) -> GroupNorm(4, 32) -> ReLU -> MaxPool(2)
      Conv1D(32 -> 64, k=3) -> GroupNorm(4, 64) -> ReLU -> MaxPool(2)
      Conv1D(64 -> 128, k=3) -> GroupNorm(4, 128) -> ReLU -> AdaptiveAvgPool(1)
      Dropout(0.2) -> Linear(128 -> 64) -> ReLU -> Dropout(0.2) -> Linear(64 -> 2)
      
    Accepts sequence windows of CAN messages: shape (Batch, Seq_Len, Features)
    or (Batch, Features, Seq_Len).
    """

    def __init__(
        self,
        in_channels: int = 11,
        seq_length: int = 16,
        num_classes: int = 2,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        fc_hidden: int = 64,
        dropout: float = 0.2,
        num_groups: Optional[int] = 4,
    ):
        super(CAN1DCNN, self).__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.num_groups = num_groups

        c1, c2, c3 = conv_channels

        # Conv Block 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn1 = get_group_norm(c1, max_groups=num_groups or 4)

        # Conv Block 2
        self.conv2 = nn.Conv1d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn2 = get_group_norm(c2, max_groups=num_groups or 4)

        # Conv Block 3
        self.conv3 = nn.Conv1d(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn3 = get_group_norm(c3, max_groups=num_groups or 4)

        # Pooling & Regularization
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)

        # Classification Head
        self.fc1 = nn.Linear(c3, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (Batch, Seq_Len, Features) or (Batch, Features, Seq_Len)
            
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # If input is (Batch, Seq_Len, Features), transpose to (Batch, Features, Seq_Len)
        if x.dim() == 3 and x.size(1) == self.seq_length and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)

        # Conv Block 1
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)

        # Conv Block 2
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)

        # Conv Block 3
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.global_pool(x)  # (Batch, c3, 1)

        # Flatten & Dense
        x = x.squeeze(-1)  # (Batch, c3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Compute trainable and total parameter counts for a PyTorch model.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
