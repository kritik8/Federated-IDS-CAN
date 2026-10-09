"""
Federated Aggregation Module for Automotive CAN IDS.
Implements standard sample-weighted FedAvg and model communication profiling.
"""

from typing import Dict, List, Tuple
import torch


def fedavg_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform sample-weighted Federated Averaging (FedAvg) across client model updates.
    
    Formula:
        w_k = n_k / sum(n_j)
        theta_global = sum(w_k * theta_k)
        
    Validation & Safety Checks:
      - Validates non-empty update list and positive total sample count.
      - Checks key consistency across all client state dictionaries.
      - Enforces shape and dtype matching for every parameter tensor.
      - Validates that aggregated entries are floating-point tensors, preventing
        silent corruption of integer buffers or incompatible state keys.
        
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        
    Returns:
        Aggregated global model state dictionary.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates list.")

    num_clients = len(client_updates)
    total_samples = sum(num_samples for _, num_samples in client_updates)
    if total_samples <= 0:
        raise ValueError(f"Total client samples must be positive, got {total_samples}")

    reference_state, _ = client_updates[0]
    param_keys = list(reference_state.keys())

    # Compatibility check across all clients
    for idx, (client_state, n_samples) in enumerate(client_updates):
        if n_samples <= 0:
            raise ValueError(f"Client at index {idx} has invalid sample count: {n_samples}")
        if set(client_state.keys()) != set(param_keys):
            missing = set(param_keys) - set(client_state.keys())
            extra = set(client_state.keys()) - set(param_keys)
            raise ValueError(
                f"Client at index {idx} state_dict mismatch. Missing: {missing}, Extra: {extra}"
            )

    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        ref_tensor = reference_state[key]
        if not torch.is_floating_point(ref_tensor):
            raise TypeError(
                f"Parameter '{key}' has non-floating dtype {ref_tensor.dtype}; "
                "FedAvg only aggregates continuous floating-point weights."
            )

        # Initialize accumulator with zero tensor on matching device and dtype
        weighted_sum = torch.zeros_like(ref_tensor, dtype=ref_tensor.dtype, device=ref_tensor.device)

        for client_idx, (client_state, n_samples) in enumerate(client_updates):
            client_tensor = client_state[key]
            if client_tensor.shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}' at client {client_idx}: "
                    f"expected {ref_tensor.shape}, got {client_tensor.shape}"
                )
            if client_tensor.dtype != ref_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch for parameter '{key}' at client {client_idx}: "
                    f"expected {ref_tensor.dtype}, got {client_tensor.dtype}"
                )

            weight = float(n_samples) / float(total_samples)
            weighted_sum += weight * client_tensor.to(device=ref_tensor.device)

        aggregated_state[key] = weighted_sum

    return aggregated_state


def estimate_model_size_bytes(model_state: Dict[str, torch.Tensor]) -> int:
    """
    Compute total serialized raw parameter size in bytes.
    
    Args:
        model_state: Model state dictionary.
        
    Returns:
        Total size in bytes.
    """
    total_bytes = 0
    for tensor in model_state.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes


def calculate_round_communication_bytes(
    model_state: Dict[str, torch.Tensor],
    num_participating_clients: int,
) -> Dict[str, float]:
    """
    Calculate estimated communication footprint for one federated training round.
    
    Assumptions:
      - Full model parameter transmission per participating client.
      - Downlink: Server broadcasts global model to each participating client.
      - Uplink: Each participating client uploads local model updates to server.
      
    Args:
        model_state: Model state dictionary.
        num_participating_clients: Number of clients in the round.
        
    Returns:
        Dict with model_size_bytes, downlink_bytes, uplink_bytes, total_round_bytes, total_round_kb, total_round_mb.
    """
    model_bytes = estimate_model_size_bytes(model_state)
    downlink_bytes = model_bytes * num_participating_clients
    uplink_bytes = model_bytes * num_participating_clients
    total_bytes = downlink_bytes + uplink_bytes

    return {
        "model_size_bytes": float(model_bytes),
        "model_size_kb": round(model_bytes / 1024.0, 2),
        "downlink_bytes": float(downlink_bytes),
        "uplink_bytes": float(uplink_bytes),
        "total_round_bytes": float(total_bytes),
        "total_round_kb": round(total_bytes / 1024.0, 2),
        "total_round_mb": round(total_bytes / (1024.0 * 1024.0), 4),
    }
