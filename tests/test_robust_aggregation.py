"""
Unit tests for Robust Aggregation Rules and Threat Model (Phase 3).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.cnn1d import CAN1DCNN
from src.federated.aggregation import (
    fedavg_aggregate,
    coordinate_median_aggregate,
    coordinate_trimmed_mean_aggregate,
    aggregate_updates,
)
from src.federated.threat import (
    apply_label_flipping,
    corrupt_model_update_delta,
    MaliciousFederatedClient,
)
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


class TestRobustAggregation(unittest.TestCase):
    """Test suite for coordinate-wise median, trimmed mean, and input validation."""

    def test_coordinate_median_odd_clients(self):
        """Verify coordinate median with odd number of clients (K=5)."""
        c_updates = [
            ({"weight": torch.tensor([1.0, 20.0], dtype=torch.float32)}, 100),
            ({"weight": torch.tensor([100.0, -10.0], dtype=torch.float32)}, 100),
            ({"weight": torch.tensor([3.0, 0.0], dtype=torch.float32)}, 100),
            ({"weight": torch.tensor([-50.0, 50.0], dtype=torch.float32)}, 100),
            ({"weight": torch.tensor([5.0, 10.0], dtype=torch.float32)}, 100),
        ]
        agg = coordinate_median_aggregate(c_updates)
        expected = torch.tensor([3.0, 10.0], dtype=torch.float32)
        torch.testing.assert_close(agg["weight"], expected)

    def test_coordinate_median_even_clients(self):
        """Verify coordinate median with even number of clients (K=4, interpolating midpoint)."""
        c_updates = [
            ({"w": torch.tensor([1.0, 10.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([2.0, 20.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([3.0, 30.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([4.0, 40.0], dtype=torch.float32)}, 10),
        ]
        agg = coordinate_median_aggregate(c_updates)
        expected = torch.tensor([2.5, 25.0], dtype=torch.float32)
        torch.testing.assert_close(agg["w"], expected)

    def test_coordinate_trimmed_mean_outlier_rejection(self):
        """Verify coordinate trimmed mean rejects extreme outliers (K=5, trim m=1)."""
        c_updates = [
            ({"w": torch.tensor([1.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([2.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([3.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([4.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([100.0], dtype=torch.float32)}, 10),
        ]
        agg = coordinate_trimmed_mean_aggregate(c_updates, trim_ratio=0.2)
        expected = torch.tensor([3.0], dtype=torch.float32)
        torch.testing.assert_close(agg["w"], expected)

    def test_trimmed_mean_validation_and_errors(self):
        """Verify trimmed mean validates trimming fraction and client counts."""
        c_updates = [
            ({"w": torch.tensor([1.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([2.0], dtype=torch.float32)}, 10),
        ]
        with self.assertRaises(ValueError):
            coordinate_trimmed_mean_aggregate(c_updates, trim_ratio=0.5)

        with self.assertRaises(ValueError):
            coordinate_trimmed_mean_aggregate(c_updates, trim_ratio=-0.1)

    def test_aggregate_updates_dispatcher(self):
        """Verify aggregate_updates routes correctly and rejects unknown methods."""
        c_updates = [
            ({"w": torch.tensor([2.0], dtype=torch.float32)}, 10),
            ({"w": torch.tensor([4.0], dtype=torch.float32)}, 10),
        ]
        agg_fedavg = aggregate_updates(c_updates, method="fedavg")
        torch.testing.assert_close(agg_fedavg["w"], torch.tensor([3.0]))

        agg_med = aggregate_updates(c_updates, method="median")
        torch.testing.assert_close(agg_med["w"], torch.tensor([3.0]))

        with self.assertRaises(ValueError):
            aggregate_updates(c_updates, method="unknown_aggregator")


class TestThreatModel(unittest.TestCase):
    """Test suite for malicious client behaviors (label flipping, update corruption)."""

    def setUp(self):
        self.seed = 42
        rng = np.random.RandomState(self.seed)
        self.y_clean = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        self.X_clean = rng.randn(5, 16, 11).astype(np.float32)

    def test_label_flipping_functionality(self):
        """Verify binary label flipping maps 0 -> 1 and 1 -> 0 exactly."""
        y_flipped, stats = apply_label_flipping(self.y_clean)

        expected = np.array([1, 1, 1, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(y_flipped, expected)
        self.assertEqual(stats["num_samples"], 5)
        self.assertEqual(stats["original_normal"], 3)
        self.assertEqual(stats["original_attack"], 2)
        self.assertEqual(stats["flipped_normal"], 2)
        self.assertEqual(stats["flipped_attack"], 3)
        self.assertEqual(stats["flipped_percentage"], 100.0)

    def test_malicious_client_label_flipping_isolation(self):
        """Verify that label flipping is applied only to malicious client, leaving benign untouched."""
        benign = FederatedClient(client_id=0, X=self.X_clean, y=self.y_clean)
        malicious = MaliciousFederatedClient(client_id=1, X=self.X_clean, y=self.y_clean, attack_type="label_flipping")

        benign_targets = [int(y.item()) for _, y in benign.dataloader.dataset]
        self.assertEqual(benign_targets, [0, 0, 0, 1, 1])

        malicious_targets = [int(y.item()) for _, y in malicious.dataloader.dataset]
        self.assertEqual(malicious_targets, [1, 1, 1, 0, 0])

    def test_model_update_corruption_delta_scaling(self):
        """Verify delta sign-reversal and scaling: theta_corrupted = theta_global - gamma * Delta."""
        global_params = {"weight": torch.tensor([1.0, 1.0], dtype=torch.float32)}
        local_params = {"weight": torch.tensor([1.5, 2.0], dtype=torch.float32)}

        corrupted = corrupt_model_update_delta(
            local_params=local_params,
            global_params=global_params,
            scaling_factor=2.0,
        )
        expected = torch.tensor([0.0, -1.0], dtype=torch.float32)
        torch.testing.assert_close(corrupted["weight"], expected)

    def test_malicious_client_update_corruption_execution(self):
        """Verify that MaliciousFederatedClient applies corruption during get_parameters()."""
        malicious = MaliciousFederatedClient(
            client_id=1,
            X=self.X_clean,
            y=self.y_clean,
            attack_type="model_update_corruption",
            scaling_factor=1.0,
        )

        global_params = malicious.get_parameters()
        malicious.set_parameters(global_params)
        params = malicious.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn("conv1.weight", params)

    def test_end_to_end_server_with_robust_aggregation(self):
        """Verify end-to-end 1-round training with robust aggregators."""
        X_synth = np.random.randn(100, 16, 11).astype(np.float32)
        y_synth = np.array([0] * 60 + [1] * 40, dtype=np.int64)

        for agg_mode in ["fedavg", "median", "trimmed_mean"]:
            server = FederatedServer(
                val_data=(X_synth[:20], y_synth[:20]),
                seed=42,
                aggregation_method=agg_mode,
            )
            clients = [
                FederatedClient(client_id=0, X=X_synth[20:60], y=y_synth[20:60], batch_size=32),
                FederatedClient(client_id=1, X=X_synth[60:100], y=y_synth[60:100], batch_size=32),
            ]
            server.register_clients(clients)
            res = server.fit(num_rounds=1, local_epochs=1, verbose=False)
            self.assertEqual(res["best_round"], 1)
            self.assertIn("val_f1", server.history)


if __name__ == "__main__":
    unittest.main()
