"""
Threat Model and Adversarial Client Module for Federated CAN Intrusion Detection.

Implements controlled Byzantine attack models for simulated vehicular clients:
  1. Attack A: Label-Flipping Attack (Data Poisoning)
  2. Attack B: Model-Update Corruption Attack (Model Poisoning via Delta Scaling / Sign Reversal)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import torch

from .client import FederatedClient


def apply_label_flipping(
    y: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply binary label-flipping to a training label array: y -> 1 - y.
    
    Mapping:
      - Normal (0) -> Attack (1)
      - Attack (1) -> Normal (0)
      
    Args:
        y: 1D numpy array of binary labels (0 or 1).
        
    Returns:
        Tuple of:
          - y_flipped: New 1D array with inverted binary labels.
          - stats: Dictionary logging original counts, flipped counts, and percentage.
    """
    unique_vals = set(np.unique(y))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Labels must be binary in {{0, 1}}, found: {unique_vals}")

    y_flipped = 1 - y
    num_total = len(y)
    num_normal_orig = int(np.sum(y == 0))
    num_attack_orig = int(np.sum(y == 1))

    stats = {
        "num_samples": num_total,
        "original_normal": num_normal_orig,
        "original_attack": num_attack_orig,
        "flipped_normal": int(np.sum(y_flipped == 0)),
        "flipped_attack": int(np.sum(y_flipped == 1)),
        "flipped_percentage": 100.0,
    }
    return y_flipped, stats


def corrupt_model_update_delta(
    local_params: Dict[str, torch.Tensor],
    global_params: Dict[str, torch.Tensor],
    scaling_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Corrupt a client's model update via delta sign-reversal and scaling.
    
    Formula:
      Delta_k = theta_k - theta_global
      Delta_k_corrupted = - scaling_factor * Delta_k
      theta_k_corrupted = theta_global + Delta_k_corrupted
                         = theta_global - scaling_factor * (theta_k - theta_global)
                         
    When scaling_factor = 1.0, this is an exact sign-reversal (anti-gradient) attack.
    When scaling_factor > 1.0, it amplifies the malicious update direction.
    
    Args:
        local_params: State dictionary of trained local client model.
        global_params: State dictionary of broadcast global model before local training.
        scaling_factor: Non-negative multiplier for the inverted delta (default 1.0).
        
    Returns:
        Corrupted state dictionary.
    """
    if scaling_factor < 0.0:
        raise ValueError(f"scaling_factor must be non-negative, got {scaling_factor}")

    corrupted_params: Dict[str, torch.Tensor] = {}

    for k, local_val in local_params.items():
        if k not in global_params:
            raise KeyError(f"Key '{k}' in local parameters not found in global parameters.")

        global_val = global_params[k].to(device=local_val.device, dtype=local_val.dtype)
        delta = local_val - global_val
        corrupted_val = global_val - scaling_factor * delta
        corrupted_params[k] = corrupted_val.clone().detach().cpu()

    return corrupted_params


class MaliciousFederatedClient(FederatedClient):
    """
    Simulated adversarial client capable of executing Byzantine data or model poisoning.
    
    Supports:
      - 'label_flipping': Inverts local binary training labels (y -> 1 - y).
      - 'model_update_corruption': Completes normal local training on clean data, then
                                   corrupts the transmitted update delta relative to the
                                   broadcast global parameters.
      - 'benign' / None: Operates as a standard honest client.
      
    Attributes:
        attack_type: 'label_flipping', 'model_update_corruption', or None.
        scaling_factor: Multiplier for model-update corruption (default 1.0).
        attack_stats: Dictionary recording attack execution diagnostics.
    """

    def __init__(
        self,
        client_id: Union[int, str],
        X: np.ndarray,
        y: np.ndarray,
        attack_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        model_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
    ):
        self.attack_type = attack_type
        self.scaling_factor = scaling_factor
        self.attack_stats: Dict[str, Any] = {
            "client_id": client_id,
            "is_malicious": attack_type is not None and attack_type != "benign",
            "attack_type": attack_type,
            "scaling_factor": scaling_factor,
        }

        # Apply label flipping before initializing DataLoader if requested
        if self.attack_type == "label_flipping":
            y_train_effective, flip_stats = apply_label_flipping(y)
            self.attack_stats["label_flip_details"] = flip_stats
        else:
            y_train_effective = y

        super().__init__(
            client_id=client_id,
            X=X,
            y=y_train_effective,
            model_config=model_config,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

        # Cache for broadcast global parameters needed for model update corruption
        self.cached_global_params: Optional[Dict[str, torch.Tensor]] = None

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Cache broadcast global parameters and reset local model weights.
        """
        super().set_parameters(parameters)
        # Store detached CPU clone of broadcast global parameters for delta computation
        self.cached_global_params = {
            k: v.clone().detach().cpu()
            for k, v in parameters.items()
        }

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return local parameters, corrupting them if attack_type == 'model_update_corruption'.
        """
        local_params = super().get_parameters()

        if self.attack_type == "model_update_corruption":
            if self.cached_global_params is None:
                # If no global parameters have been broadcast yet, return uncorrupted initial weights
                return local_params
            corrupted = corrupt_model_update_delta(
                local_params=local_params,
                global_params=self.cached_global_params,
                scaling_factor=self.scaling_factor,
            )
            return corrupted

        return local_params
