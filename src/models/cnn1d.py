"""
Lightweight 1D-CNN Model for Automotive CAN Bus Intrusion Detection
Optimized for edge in-vehicle execution and Federated Learning aggregation.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAN1DCNN(nn.Module):
    """
    Lightweight 1D Convolutional Neural Network for CAN Bus Intrusion Detection.
    
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
    ):
        super(CAN1DCNN, self).__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.num_classes = num_classes

        c1, c2, c3 = conv_channels

        # Conv Block 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(c1)

        # Conv Block 2
        self.conv2 = nn.Conv1d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(c2)

        # Conv Block 3
        self.conv3 = nn.Conv1d(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(c3)

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
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
