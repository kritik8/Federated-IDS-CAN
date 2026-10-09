"""
Federated Server and Orchestration Module for Automotive CAN Bus IDS.
Manages global model broadcast, client coordination, FedAvg aggregation, and validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.cnn1d import CAN1DCNN, count_parameters
from ..evaluation.metrics import compute_classification_metrics, plot_training_curves, plot_confusion_matrix
from .aggregation import fedavg_aggregate, aggregate_updates, calculate_round_communication_bytes
from .client import FederatedClient


class FederatedServer:
    """
    Central server orchestrating federated learning rounds.
    
    Attributes:
        device: torch device ('cpu' or 'cuda').
        global_model: Master CAN1DCNN instance.
        val_loader: DataLoader for global held-out validation set.
        test_loader: DataLoader for global held-out test set.
        clients: Registered list of FederatedClient instances.
        history: Dictionary tracking round-by-round metrics.
    """

    def __init__(
        self,
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        eval_batch_size: int = 128,
        seed: int = 42,
        aggregation_method: str = "fedavg",
        aggregation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = torch.device(device)
        self.eval_batch_size = eval_batch_size
        self.seed = seed
        self.aggregation_method = aggregation_method
        self.aggregation_kwargs = aggregation_kwargs or {}
        self.model_config = model_config or {}

        # Set seeds for deterministic server initialization
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize global master model
        self.global_model = CAN1DCNN(
            in_channels=self.model_config.get("in_channels", 11),
            seq_length=self.model_config.get("seq_length", 16),
            num_classes=self.model_config.get("num_classes", 2),
            conv_channels=self.model_config.get("conv_channels", (32, 64, 128)),
            fc_hidden=self.model_config.get("fc_hidden", 64),
            dropout=self.model_config.get("dropout", 0.2),
            num_groups=self.model_config.get("num_groups", 4),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Build validation loader
        val_x, val_y = val_data
        self.val_dataset = TensorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.long),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )

        # Build test loader if provided
        self.test_loader: Optional[DataLoader] = None
        if test_data is not None:
            test_x, test_y = test_data
            self.test_dataset = TensorDataset(
                torch.tensor(test_x, dtype=torch.float32),
                torch.tensor(test_y, dtype=torch.long),
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
            )

        self.clients: List[FederatedClient] = []
        self.history: Dict[str, List[Any]] = {
            "round": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "round_duration_seconds": [],
            "communication": [],
        }

        self.best_round: int = -1
        self.best_val_f1: float = -1.0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None

    def register_clients(self, clients: List[FederatedClient]) -> None:
        """Register the participating federated clients with the server."""
        self.clients = clients

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Return detached CPU copy of global model state."""
        return {
            k: v.clone().detach().cpu()
            for k, v in self.global_model.state_dict().items()
        }

    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Load aggregated parameters into global model."""
        cloned_state = {
            k: v.clone().detach().to(self.device)
            for k, v in parameters.items()
        }
        self.global_model.load_state_dict(cloned_state)

    def evaluate_global(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate current global model on a given evaluation DataLoader.
        
        Args:
            dataloader: DataLoader with TensorDataset(X, y).
            
        Returns:
            Tuple of (mean_loss, classification_metrics_dict).
        """
        self.global_model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for x_b, y_b in dataloader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                logits = self.global_model(x_b)
                loss = self.criterion(logits, y_b)

                running_loss += loss.item() * len(y_b)
                total_samples += len(y_b)

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(y_b.cpu().numpy())
                all_probs.extend(probs)

        mean_loss = float(running_loss / total_samples) if total_samples > 0 else 0.0
        metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
        )
        return mean_loss, metrics

    def fit(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        verbose: bool = True,
        aggregation_method: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the complete federated training loop across rounds.
        
        Each round:
          1. Broadcasts identical global parameters to all clients.
          2. Each client trains locally for `local_epochs`.
          3. Collects updated client parameters and sample counts.
          4. Aggregates updates via the configured aggregation rule (FedAvg, Median, Trimmed Mean).
          5. Updates master global model.
          6. Evaluates global model on untouched validation set.
          7. Tracks best validation F1 checkpoint.
          
        Args:
            num_rounds: Number of federated communication rounds.
            local_epochs: Number of local epochs each client performs per round.
            verbose: If True, prints round-by-round progress.
            aggregation_method: Optional override for aggregation algorithm.
            **kwargs: Extra parameters for aggregation (e.g. trim_ratio).
            
        Returns:
            Dictionary containing training history and best round metadata.
        """
        if not self.clients:
            raise RuntimeError("No clients registered. Call register_clients() first.")

        agg_method = (aggregation_method or self.aggregation_method).lower().strip()
        agg_kwargs = {**self.aggregation_kwargs, **kwargs}

        if verbose:
            print("=" * 70)
            print(f"STARTING FEDERATED TRAINING: {num_rounds} Rounds, {len(self.clients)} Clients ({agg_method})")
            print(f"Device: {self.device} | Local Epochs/Round: {local_epochs} | Seed: {self.seed}")
            print("=" * 70)

        total_training_start = time.perf_counter()

        for round_idx in range(1, num_rounds + 1):
            t_round_start = time.perf_counter()

            # 1. Broadcast global parameters to all participating clients
            current_global_params = self.get_global_parameters()
            for client in self.clients:
                client.set_parameters(current_global_params)

            # 2. Local client training
            client_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
            client_losses: List[float] = []

            for client in self.clients:
                client_res = client.train_epoch(epochs=local_epochs)
                client_losses.append(client_res["train_loss"])
                client_updates.append((client.get_parameters(), client.get_sample_count()))

            mean_train_loss = float(np.mean(client_losses))

            # 3. Federated Aggregation via configured rule
            aggregated_params = aggregate_updates(
                client_updates,
                method=agg_method,
                **agg_kwargs,
            )

            # 4. Update master global model
            self.set_global_parameters(aggregated_params)

            # 5. Evaluate updated global model on held-out validation set
            val_loss, val_metrics = self.evaluate_global(self.val_loader)
            round_duration = time.perf_counter() - t_round_start

            # 6. Estimate communication cost
            comm_stats = calculate_round_communication_bytes(
                aggregated_params,
                num_participating_clients=len(self.clients),
            )

            # 7. Record round history
            val_f1 = float(val_metrics["f1_score"])
            self.history["round"].append(round_idx)
            self.history["train_loss"].append(mean_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(float(val_metrics["accuracy"]))
            self.history["val_precision"].append(float(val_metrics["precision"]))
            self.history["val_recall"].append(float(val_metrics["recall"]))
            self.history["val_f1"].append(val_f1)
            self.history["round_duration_seconds"].append(round(round_duration, 4))
            self.history["communication"].append(comm_stats)

            # 8. Track best validation F1 checkpoint
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_round = round_idx
                self.best_model_state = copy.deepcopy(aggregated_params)

            if verbose:
                print(
                    f"Round [{round_idx:2d}/{num_rounds:2d}] | "
                    f"Train Loss: {mean_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
                    f"Time: {round_duration:.2f}s"
                )

        total_training_duration = time.perf_counter() - total_training_start

        if verbose:
            print("=" * 70)
            print(f"FEDERATED TRAINING COMPLETE in {total_training_duration:.2f}s")
            print(f"Selected Best Checkpoint: Round {self.best_round} (Val F1: {self.best_val_f1 * 100:.2f}%)")
            print("=" * 70)

        # Restore best checkpoint into global model for subsequent test evaluation
        if self.best_model_state is not None:
            self.set_global_parameters(self.best_model_state)

        return {
            "total_duration_seconds": round(total_training_duration, 2),
            "best_round": self.best_round,
            "best_val_f1": self.best_val_f1,
            "history": self.history,
        }

    def evaluate_test(self) -> Dict[str, Any]:
        """
        Evaluate the selected best global model once on the untouched test set.
        """
        if self.test_loader is None:
            raise RuntimeError("Test data loader not initialized.")

        # Ensure model is evaluated in best validation checkpoint state
        if self.best_model_state is not None:
            self.set_global_parameters(self.best_model_state)

        test_loss, test_metrics = self.evaluate_global(self.test_loader)
        test_metrics["test_loss"] = test_loss
        return test_metrics
