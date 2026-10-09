"""
Federated Client Module for Automotive CAN Bus Intrusion Detection.
Simulates edge vehicular ECU local training with CAN1DCNN.
"""

from typing import Any, Dict, Optional, Tuple, Union
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.cnn1d import CAN1DCNN


class FederatedClient:
    """
    Simulated vehicular client that performs local intrusion detection model training.
    
    Attributes:
        client_id: Identifier for client (e.g. 0 to 9).
        num_samples: Total number of local training samples.
        device: torch device ('cpu' or 'cuda').
        batch_size: Batch size for local DataLoader.
        lr: Local learning rate.
        weight_decay: L2 regularization parameter.
    """

    def __init__(
        self,
        client_id: Union[int, str],
        X: np.ndarray,
        y: np.ndarray,
        model_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
    ):
        self.client_id = client_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_samples = len(y)

        if self.num_samples == 0:
            raise ValueError(f"Client {client_id} cannot be initialized with 0 samples.")

        # Class distribution diagnostics
        self.num_normal = int(np.sum(y == 0))
        self.num_attack = int(np.sum(y == 1))

        # Setup local DataLoader
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        self.dataset = TensorDataset(tensor_x, tensor_y)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Instantiate local model
        cfg = model_config or {}
        self.model = CAN1DCNN(
            in_channels=cfg.get("in_channels", 11),
            seq_length=cfg.get("seq_length", 16),
            num_classes=cfg.get("num_classes", 2),
            conv_channels=cfg.get("conv_channels", (32, 64, 128)),
            fc_hidden=cfg.get("fc_hidden", 64),
            dropout=cfg.get("dropout", 0.2),
            num_groups=cfg.get("num_groups", 4),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Reset local model parameters to broadcast global model state.
        
        Args:
            parameters: State dictionary of global model parameters.
        """
        cloned_state = {
            k: v.clone().detach().to(self.device)
            for k, v in parameters.items()
        }
        self.model.load_state_dict(cloned_state)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return a detached CPU copy of current local model parameters.
        """
        return {
            k: v.clone().detach().cpu()
            for k, v in self.model.state_dict().items()
        }

    def train_epoch(self, epochs: int = 1) -> Dict[str, Any]:
        """
        Train local model for the specified number of local epochs on client data.
        
        Args:
            epochs: Number of local training epochs (default 1).
            
        Returns:
            Dict containing local training metrics (loss, duration, sample count).
        """
        self.model.train()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        t0 = time.perf_counter()
        epoch_losses = []

        for _ in range(epochs):
            running_loss = 0.0
            for x_b, y_b in self.dataloader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_b)
                loss = self.criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(y_b)

            epoch_loss = running_loss / self.num_samples
            epoch_losses.append(epoch_loss)

        duration = time.perf_counter() - t0
        mean_loss = float(np.mean(epoch_losses))

        return {
            "client_id": self.client_id,
            "train_loss": mean_loss,
            "num_samples": self.num_samples,
            "local_epochs": epochs,
            "duration_seconds": round(duration, 4),
        }

    def get_sample_count(self) -> int:
        """Return total number of local training samples."""
        return self.num_samples

    def get_class_distribution(self) -> Dict[str, int]:
        """Return dictionary of local class counts."""
        return {
            "normal": self.num_normal,
            "attack": self.num_attack,
        }
