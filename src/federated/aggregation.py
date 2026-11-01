"""
Federated Aggregation Module for Automotive CAN IDS.
Implements:
  1. Standard sample-weighted FedAvg (McMahan et al., 2017)
  2. Coordinate-wise Median (Yin et al., 2018)
  3. Coordinate-wise Trimmed Mean (Yin et al., 2018)
  4. Analytical model communication footprint estimation.

Translation Equivariance Note:
  For any coordinate-wise aggregation rule (FedAvg, Median, Trimmed Mean):
    Agg({theta_k}) = theta_global + Agg({theta_k - theta_global})
  Aggregating full parameter states directly is mathematically identical to
  aggregating delta updates and adding to the broadcast global model.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def validate_client_updates(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> List[str]:
    """
    Validate safety, key consistency, shapes, and floating-point dtypes across client updates.
    
    Args:
        client_updates: List of (state_dict, num_samples) tuples.
        
    Returns:
        List of verified parameter keys.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates list.")

    reference_state, _ = client_updates[0]
    param_keys = list(reference_state.keys())

    for idx, (client_state, n_samples) in enumerate(client_updates):
        if n_samples <= 0:
            raise ValueError(f"Client at index {idx} has invalid sample count: {n_samples}")
        if set(client_state.keys()) != set(param_keys):
            missing = set(param_keys) - set(client_state.keys())
            extra = set(client_state.keys()) - set(param_keys)
            raise ValueError(
                f"Client at index {idx} state_dict mismatch. Missing: {missing}, Extra: {extra}"
            )

        for key in param_keys:
            ref_tensor = reference_state[key]
            client_tensor = client_state[key]

            if not torch.is_floating_point(ref_tensor):
                raise TypeError(
                    f"Parameter '{key}' has non-floating dtype {ref_tensor.dtype}; "
                    "aggregators only aggregate continuous floating-point weights."
                )
            if client_tensor.shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}' at client {idx}: "
                    f"expected {ref_tensor.shape}, got {client_tensor.shape}"
                )
            if client_tensor.dtype != ref_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch for parameter '{key}' at client {idx}: "
                    f"expected {ref_tensor.dtype}, got {client_tensor.dtype}"
                )

    return param_keys


def fedavg_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform sample-weighted Federated Averaging (FedAvg) across client model updates.
    
    Formula:
        w_k = n_k / sum(n_j)
        theta_global = sum(w_k * theta_k)
        
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        
    Returns:
        Aggregated global model state dictionary.
    """
    param_keys = validate_client_updates(client_updates)
    total_samples = sum(num_samples for _, num_samples in client_updates)

    reference_state, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        ref_tensor = reference_state[key]
        weighted_sum = torch.zeros_like(ref_tensor, dtype=ref_tensor.dtype, device=ref_tensor.device)

        for client_state, n_samples in client_updates:
            client_tensor = client_state[key].to(device=ref_tensor.device)
            weight = float(n_samples) / float(total_samples)
            weighted_sum += weight * client_tensor

        aggregated_state[key] = weighted_sum

    return aggregated_state


def coordinate_median_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform Coordinate-wise Median aggregation across client model updates.
    
    For each coordinate j:
        theta_global[j] = median({theta_k[j] : k = 1, ..., K})
        
    When K is even, computes the standard arithmetic mean of the two central values.
    
    Note on Robust Estimation:
      Following standard Byzantine-robust learning literature (Yin et al., 2018),
      robust estimators evaluate coordinates without sample weighting, preventing
      malicious clients from subverting aggregation by claiming inflated sample counts.
      
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        
    Returns:
        Aggregated global model state dictionary.
    """
    param_keys = validate_client_updates(client_updates)
    reference_state, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        ref_tensor = reference_state[key]
        # Stack all K client tensors along dimension 0: shape (K, *tensor_shape)
        stacked = torch.stack(
            [cs[key].to(device=ref_tensor.device) for cs, _ in client_updates],
            dim=0,
        )
        # Compute coordinate-wise median using quantile(q=0.5) to interpolate middle values for even K
        median_tensor = torch.quantile(stacked, q=0.5, dim=0)
        aggregated_state[key] = median_tensor

    return aggregated_state


def coordinate_trimmed_mean_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    trim_ratio: float = 0.2,
    num_trim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Perform Coordinate-wise Trimmed Mean aggregation across client model updates.
    
    For each coordinate j, sorts the K client values:
        v_(1) <= v_(2) <= ... <= v_(K)
    Trims the m smallest and m largest values, and averages the remaining K - 2m values:
        theta_global[j] = (1 / (K - 2m)) * sum_{i=m+1}^{K-m} v_(i)
        
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        trim_ratio: Trimming fraction in [0.0, 0.5) (default 0.2).
        num_trim: Optional explicit integer count of extreme values to trim per side.
                  If provided, overrides trim_ratio.
                  
    Returns:
        Aggregated global model state dictionary.
    """
    param_keys = validate_client_updates(client_updates)
    num_clients = len(client_updates)

    if num_trim is not None:
        m = int(num_trim)
    else:
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError(f"trim_ratio must be in [0.0, 0.5), got {trim_ratio}")
        m = int(num_clients * trim_ratio)

    if m < 0:
        raise ValueError(f"Trimming count m must be non-negative, got {m}")
    if 2 * m >= num_clients:
        raise ValueError(
            f"Cannot trim {2*m} updates from {num_clients} clients: "
            f"2*m ({2*m}) must be strictly less than K ({num_clients}). "
            f"Retained updates: {num_clients - 2*m} <= 0."
        )

    reference_state, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in param_keys:
        ref_tensor = reference_state[key]
        stacked = torch.stack(
            [cs[key].to(device=ref_tensor.device) for cs, _ in client_updates],
            dim=0,
        )
        sorted_stacked, _ = torch.sort(stacked, dim=0)

        if m > 0:
            retained = sorted_stacked[m : num_clients - m]
        else:
            retained = sorted_stacked

        trimmed_mean_tensor = retained.mean(dim=0)
        aggregated_state[key] = trimmed_mean_tensor

    return aggregated_state


def multi_krum_aggregate(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    num_byzantine: int = 2,
    num_selected: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Perform Multi-Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).
    
    Mathematical Formulation:
      Let n be the total number of participating clients, and f be the maximum number
      of Byzantine (compromised) clients.
      
      Standard theoretical feasibility condition:
          n >= 2f + 3
          
      For each client update i in {1, ..., n}:
        1. Flatten all continuous floating-point parameters into a 1D vector v_i.
        2. Compute pairwise squared Euclidean distances: d(i, j) = ||v_i - v_j||_2^2.
        3. Exclude self-distance (d(i, i) = 0).
        4. Calculate Krum score s_i by summing distances to its (n - f - 2) nearest neighbors:
             s_i = sum_{j in N_i} d(i, j), where |N_i| = n - f - 2.
        5. Rank clients by s_i in ascending order.
        6. Select the m updates with the lowest Krum scores, where by default:
             m = n - f - 2
        7. Aggregate the selected updates using their arithmetic mean:
             theta_global = (1 / m) * sum_{i in S} theta_i.
             
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        num_byzantine: Assumed upper bound on Byzantine clients f (default 2).
        num_selected: Number of candidate updates m to select (default n - f - 2).
        return_diagnostics: If True, returns (aggregated_state, diagnostics_dict).
        
    Returns:
        Aggregated model state dictionary (or tuple with diagnostics).
    """
    param_keys = validate_client_updates(client_updates)
    n = len(client_updates)
    f = int(num_byzantine)

    if f < 0:
        raise ValueError(f"num_byzantine must be non-negative, got {f}")

    # Standard Multi-Krum feasibility condition: n >= 2f + 3
    if n < 2 * f + 3:
        raise ValueError(
            f"Multi-Krum requires n >= 2f + 3. Got n={n} clients and f={f} Byzantine budget. "
            f"Minimum required clients for f={f} is {2 * f + 3}."
        )

    # Standard Multi-Krum selection parameter: m = n - f - 2
    if num_selected is not None:
        m = int(num_selected)
    else:
        m = n - f - 2

    if m < 1 or m > n:
        raise ValueError(f"Number of selected updates m must satisfy 1 <= m <= n ({n}), got {m}")

    reference_state, _ = client_updates[0]

    # 1. Flatten client parameters into 1D vectors
    flat_vectors = []
    for client_state, _ in client_updates:
        flat = torch.cat([
            client_state[k].detach().flatten().to(dtype=torch.float32, device="cpu")
            for k in param_keys
        ])
        flat_vectors.append(flat)

    stacked_vectors = torch.stack(flat_vectors, dim=0)  # Shape: (n, D)

    # 2. Compute pairwise squared Euclidean distances
    # Shape: (n, n)
    dists = torch.cdist(stacked_vectors, stacked_vectors, p=2.0) ** 2

    # 3. Compute Krum score for each client
    # For client i, distances to all other clients sorted ascending
    # dists[i, i] is 0.0 (index 0). Indices 1 to (n - f - 2) are the (n - f - 2) nearest other updates.
    num_neighbors = n - f - 2
    scores = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        sorted_d, _ = torch.sort(dists[i])
        # Skip index 0 (self-distance)
        nearest_d = sorted_d[1 : 1 + num_neighbors]
        scores[i] = nearest_d.sum()

    # 4. Rank clients by Krum score
    ranked_indices = torch.argsort(scores).tolist()
    selected_indices = ranked_indices[:m]

    # 5. Arithmetic mean of the m selected updates
    aggregated_state: Dict[str, torch.Tensor] = {}
    for key in param_keys:
        ref_tensor = reference_state[key]
        stacked_selected = torch.stack(
            [client_updates[idx][0][key].to(device=ref_tensor.device) for idx in selected_indices],
            dim=0,
        )
        aggregated_state[key] = stacked_selected.mean(dim=0)

    if return_diagnostics:
        diagnostics = {
            "n_clients": n,
            "f_byzantine": f,
            "m_selected": m,
            "krum_scores": [float(s) for s in scores.tolist()],
            "ranked_client_indices": ranked_indices,
            "selected_client_indices": selected_indices,
        }
        return aggregated_state, diagnostics

    return aggregated_state


def aggregate_updates(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    method: str = "fedavg",
    **kwargs: Any,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Dispatcher function for federated model update aggregation.
    
    Supported methods:
      - 'fedavg': Standard sample-weighted FedAvg.
      - 'median' or 'coordinate_median': Unweighted coordinate-wise median.
      - 'trimmed_mean' or 'coordinate_trimmed_mean': Unweighted coordinate-wise trimmed mean.
      - 'multi_krum' or 'krum': Distance-based Multi-Krum aggregation.
      
    Args:
        client_updates: List of tuples (model_state_dict, num_samples).
        method: Aggregation algorithm name.
        **kwargs: Additional parameters (e.g. trim_ratio, num_trim, num_byzantine, num_selected).
        
    Returns:
        Aggregated state dictionary (or tuple with diagnostics if requested).
    """
    norm_method = method.lower().strip()

    if norm_method in ("fedavg", "sample_weighted_fedavg"):
        return fedavg_aggregate(client_updates)
    elif norm_method in ("median", "coordinate_median"):
        return coordinate_median_aggregate(client_updates)
    elif norm_method in ("trimmed_mean", "coordinate_trimmed_mean"):
        trim_ratio = kwargs.get("trim_ratio", 0.2)
        num_trim = kwargs.get("num_trim", None)
        return coordinate_trimmed_mean_aggregate(client_updates, trim_ratio=trim_ratio, num_trim=num_trim)
    elif norm_method in ("multi_krum", "multikrum", "krum"):
        num_byzantine = kwargs.get("num_byzantine", 2)
        num_selected = kwargs.get("num_selected", None)
        return_diag = kwargs.get("return_diagnostics", False)
        return multi_krum_aggregate(
            client_updates,
            num_byzantine=num_byzantine,
            num_selected=num_selected,
            return_diagnostics=return_diag,
        )
    else:
        raise ValueError(
            f"Unknown aggregation method '{method}'. Supported methods: "
            f"['fedavg', 'coordinate_median', 'coordinate_trimmed_mean', 'multi_krum']"
        )


def estimate_model_size_bytes(model_state: Dict[str, torch.Tensor]) -> int:
    """
    Compute total serialized raw parameter size in bytes.
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
