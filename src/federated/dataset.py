"""
Dataset Partitioning Module for Federated Learning on Automotive CAN Bus IDS.
Supports reproducible, leak-free IID partitioning across simulated vehicular clients.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def partition_iid_stratified(
    y: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition sample indices into balanced, stratified IID subsets across clients.
    
    Methodology:
      - For each unique class label (e.g., Normal=0, Attack=1), extract its indices.
      - Shuffle indices deterministically using a fixed random seed.
      - Split the shuffled indices into `num_clients` contiguous chunks.
      - Assign one chunk from each class to each client, guaranteeing matching class ratios.
      - Shuffle each client's combined local indices.
      
    Guarantees:
      1. Mutual Exclusivity: Every training index belongs to exactly one client.
      2. Completeness: Union of all client indices equals the full dataset.
      3. Class Balance: Client attack proportions approximately match global proportion (~39.1%).
      4. Sample Balance: Client sample sizes are balanced (e.g. 2,187 or 2,188 for N=21,875).
      5. Reproducibility: Deterministic split governed strictly by `seed`.
      
    Note on Simulation:
      These represent simulated edge vehicular ECUs / vehicle partitions operating
      over a common automotive traffic baseline, not naturally separated physical vehicles.
      
    Args:
        y: 1D array of integer labels (shape: [N,]).
        num_clients: Number of simulated clients (default 10).
        seed: Random seed for deterministic shuffling (default 42).
        
    Returns:
        List of 1D numpy integer arrays containing index allocations per client.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if len(y) < num_clients:
        raise ValueError(f"Dataset size ({len(y)}) is smaller than num_clients ({num_clients})")

    rng = np.random.RandomState(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    unique_classes = np.unique(y)
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        chunks = np.array_split(cls_indices, num_clients)
        for i, chunk in enumerate(chunks):
            client_indices[i].extend(chunk.tolist())

    # Shuffle local client indices deterministically
    final_client_indices: List[np.ndarray] = []
    for i in range(num_clients):
        local_idx = np.array(client_indices[i], dtype=np.int64)
        rng.shuffle(local_idx)
        final_client_indices.append(local_idx)

    return final_client_indices


def create_iid_client_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_types_train: Optional[np.ndarray] = None,
    num_clients: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Partition the centralized training dataset into local datasets for simulated clients.
    
    Validation and test datasets are kept strictly centralized and untouched to serve as
    the global benchmark.
    
    Args:
        X_train: Training features array of shape (N, W, F).
        y_train: Training binary labels array of shape (N,).
        y_types_train: Optional multi-class attack type strings (shape: [N,]).
        num_clients: Number of simulated vehicular clients.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of:
          - client_datasets: List of dicts, each containing:
              'client_id': int
              'X': local feature array
              'y': local binary label array
              'y_types': local attack type string array (if provided)
          - summary: Dictionary containing distribution and partition diagnostics.
    """
    client_indices = partition_iid_stratified(y_train, num_clients=num_clients, seed=seed)

    client_datasets: List[Dict[str, np.ndarray]] = []
    client_summaries: List[Dict[str, Any]] = []

    for client_id, idx in enumerate(client_indices):
        X_c = X_train[idx]
        y_c = y_train[idx]
        types_c = y_types_train[idx] if y_types_train is not None else None

        client_dict: Dict[str, Any] = {
            "client_id": client_id,
            "X": X_c,
            "y": y_c,
        }
        if types_c is not None:
            client_dict["y_types"] = types_c
        client_datasets.append(client_dict)

        num_normal = int(np.sum(y_c == 0))
        num_attack = int(np.sum(y_c == 1))
        attack_ratio = float(num_attack / len(y_c)) if len(y_c) > 0 else 0.0

        summary_entry: Dict[str, Any] = {
            "client_id": client_id,
            "num_samples": len(y_c),
            "num_normal": num_normal,
            "num_attack": num_attack,
            "attack_ratio": round(attack_ratio, 4),
        }
        if types_c is not None:
            unique_types, counts = np.unique(types_c, return_counts=True)
            summary_entry["attack_types"] = dict(zip(unique_types.tolist(), counts.tolist()))

        client_summaries.append(summary_entry)

    overall_summary: Dict[str, Any] = {
        "partition_type": "IID_stratified",
        "num_clients": num_clients,
        "seed": seed,
        "total_training_samples": len(y_train),
        "clients": client_summaries,
    }

    return client_datasets, overall_summary
