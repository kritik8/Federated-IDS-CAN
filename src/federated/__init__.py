"""
Federated Learning Package for Vehicular CAN Bus Intrusion Detection.
"""

from .dataset import partition_iid_stratified, create_iid_client_datasets
from .aggregation import (
    fedavg_aggregate,
    estimate_model_size_bytes,
    calculate_round_communication_bytes,
)
from .client import FederatedClient
from .server import FederatedServer

__all__ = [
    "partition_iid_stratified",
    "create_iid_client_datasets",
    "fedavg_aggregate",
    "estimate_model_size_bytes",
    "calculate_round_communication_bytes",
    "FederatedClient",
    "FederatedServer",
]
