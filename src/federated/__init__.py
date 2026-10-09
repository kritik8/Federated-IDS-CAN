"""
Federated Learning Package for Vehicular CAN Bus Intrusion Detection.
"""

from .dataset import (
    partition_iid_stratified,
    create_iid_client_datasets,
    partition_dirichlet,
    create_noniid_client_datasets,
)
from .aggregation import (
    fedavg_aggregate,
    coordinate_median_aggregate,
    coordinate_trimmed_mean_aggregate,
    aggregate_updates,
    estimate_model_size_bytes,
    calculate_round_communication_bytes,
)
from .client import FederatedClient
from .threat import (
    MaliciousFederatedClient,
    apply_label_flipping,
    corrupt_model_update_delta,
)
from .server import FederatedServer

__all__ = [
    "partition_iid_stratified",
    "create_iid_client_datasets",
    "partition_dirichlet",
    "create_noniid_client_datasets",
    "fedavg_aggregate",
    "coordinate_median_aggregate",
    "coordinate_trimmed_mean_aggregate",
    "aggregate_updates",
    "estimate_model_size_bytes",
    "calculate_round_communication_bytes",
    "FederatedClient",
    "MaliciousFederatedClient",
    "apply_label_flipping",
    "corrupt_model_update_delta",
    "FederatedServer",
]
