"""
Unit tests for the Federated Learning simulation module (src/federated/).
Tests both Phase 2A (IID FedAvg) and Phase 2B (Dirichlet Non-IID partitioning and FedAvg).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.models.cnn1d import CAN1DCNN
from src.federated.dataset import (
    partition_iid_stratified,
    create_iid_client_datasets,
    partition_dirichlet,
    create_noniid_client_datasets,
)
from src.federated.aggregation import (
    fedavg_aggregate,
    estimate_model_size_bytes,
    calculate_round_communication_bytes,
)
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


class TestFederatedIID(unittest.TestCase):
    """Test suite for IID partitioning, FedAvg aggregation, client and server routines."""

    def setUp(self):
        self.num_clients = 10
        self.seed = 42
        # Synthetic dataset with 1,000 samples (600 normal, 400 attack)
        self.num_samples = 1000
        rng = np.random.RandomState(self.seed)
        self.y_synth = np.array([0] * 600 + [1] * 400, dtype=np.int64)
        rng.shuffle(self.y_synth)
        self.X_synth = rng.randn(self.num_samples, 16, 11).astype(np.float32)

    def test_iid_partition_completeness_and_no_overlap(self):
        """Verify mutual exclusivity, completeness, and no duplicate indices."""
        partitions = partition_iid_stratified(self.y_synth, num_clients=self.num_clients, seed=self.seed)

        self.assertEqual(len(partitions), self.num_clients)
        all_indices = []
        for p in partitions:
            all_indices.extend(p.tolist())

        # Completeness: all samples assigned
        self.assertEqual(len(all_indices), self.num_samples)
        # Mutual exclusivity: no duplicate sample assignments
        self.assertEqual(len(set(all_indices)), self.num_samples)
        # Range check: indices match 0 .. num_samples - 1
        self.assertEqual(set(all_indices), set(range(self.num_samples)))

    def test_iid_partition_reproducibility(self):
        """Verify that partitioning with the same seed produces identical index assignments."""
        part1 = partition_iid_stratified(self.y_synth, num_clients=self.num_clients, seed=self.seed)
        part2 = partition_iid_stratified(self.y_synth, num_clients=self.num_clients, seed=self.seed)

        for p1, p2 in zip(part1, part2):
            np.testing.assert_array_equal(p1, p2)

        # Different seed should produce different partition
        part_diff = partition_iid_stratified(self.y_synth, num_clients=self.num_clients, seed=123)
        diff_found = any(not np.array_equal(p1, pd) for p1, pd in zip(part1, part_diff))
        self.assertTrue(diff_found, "Different seeds should produce different partitions.")

    def test_iid_approximate_class_balance(self):
        """Verify that client attack ratios are all approximately equal to global ratio (0.40)."""
        partitions = partition_iid_stratified(self.y_synth, num_clients=self.num_clients, seed=self.seed)
        global_ratio = float(np.mean(self.y_synth))  # 0.40

        for idx, p in enumerate(partitions):
            client_y = self.y_synth[p]
            client_ratio = float(np.mean(client_y))
            self.assertAlmostEqual(client_ratio, global_ratio, places=2)
            self.assertEqual(len(p), 100)

    def test_create_iid_client_datasets_diagnostics(self):
        """Verify create_iid_client_datasets returns well-formed dictionaries and summary stats."""
        client_datasets, summary = create_iid_client_datasets(
            self.X_synth,
            self.y_synth,
            num_clients=self.num_clients,
            seed=self.seed,
        )
        self.assertEqual(len(client_datasets), self.num_clients)
        self.assertEqual(summary["total_training_samples"], self.num_samples)
        self.assertEqual(len(summary["clients"]), self.num_clients)

        total_samples_collected = sum(c["num_samples"] for c in summary["clients"])
        self.assertEqual(total_samples_collected, self.num_samples)

    def test_fedavg_correctness_and_sample_weighting(self):
        """Verify FedAvg calculates exact sample-weighted mathematical average."""
        state_a = {"weight": torch.tensor([2.0, 4.0], dtype=torch.float32)}
        state_b = {"weight": torch.tensor([6.0, 8.0], dtype=torch.float32)}

        updates = [(state_a, 100), (state_b, 300)]
        agg = fedavg_aggregate(updates)

        expected = torch.tensor([5.0, 7.0], dtype=torch.float32)
        torch.testing.assert_close(agg["weight"], expected)

    def test_fedavg_validation_checks(self):
        """Verify fedavg_aggregate raises errors for mismatched keys or non-floating types."""
        state_a = {"w1": torch.tensor([1.0])}
        state_b = {"w2": torch.tensor([1.0])}
        with self.assertRaises(ValueError):
            fedavg_aggregate([(state_a, 10), (state_b, 10)])

        state_int = {"w1": torch.tensor([1, 2], dtype=torch.int64)}
        with self.assertRaises(TypeError):
            fedavg_aggregate([(state_int, 10)])

        with self.assertRaises(ValueError):
            fedavg_aggregate([])

    def test_identical_initialization_at_round_start(self):
        """Verify that server broadcasts identical parameters and client models are reset."""
        val_data = (self.X_synth[:50], self.y_synth[:50])
        server = FederatedServer(val_data=val_data, seed=self.seed)

        client_data = (self.X_synth[:100], self.y_synth[:100])
        client = FederatedClient(client_id=0, X=client_data[0], y=client_data[1])

        global_params = server.get_global_parameters()
        client.set_parameters(global_params)
        client_params = client.get_parameters()

        for k in global_params:
            torch.testing.assert_close(client_params[k], global_params[k])

    def test_one_round_training_end_to_end(self):
        """Verify a complete 1-round federated training run on small synthetic dataset."""
        train_x = self.X_synth[:200]
        train_y = self.y_synth[:200]
        val_x = self.X_synth[200:250]
        val_y = self.y_synth[200:250]

        client_datasets, _ = create_iid_client_datasets(train_x, train_y, num_clients=2, seed=self.seed)

        server = FederatedServer(val_data=(val_x, val_y), seed=self.seed)
        clients = [
            FederatedClient(client_id=cd["client_id"], X=cd["X"], y=cd["y"], batch_size=32)
            for cd in client_datasets
        ]
        server.register_clients(clients)

        res = server.fit(num_rounds=1, local_epochs=1, verbose=False)

        self.assertEqual(res["best_round"], 1)
        self.assertEqual(len(server.history["round"]), 1)
        self.assertGreaterEqual(server.history["val_accuracy"][0], 0.0)
        self.assertLessEqual(server.history["val_accuracy"][0], 1.0)

    def test_communication_cost_estimation(self):
        """Verify communication cost calculation matches known 40,610 param model footprint."""
        model = CAN1DCNN(in_channels=11, seq_length=16, num_classes=2)
        comm = calculate_round_communication_bytes(model.state_dict(), num_participating_clients=10)

        self.assertEqual(comm["model_size_bytes"], 162440)
        self.assertAlmostEqual(comm["model_size_kb"], 158.63, places=1)
        self.assertEqual(comm["total_round_bytes"], 162440 * 20)


class TestFederatedNonIID(unittest.TestCase):
    """Test suite for Dirichlet Non-IID partitioning, metadata handling, and skew properties."""

    def setUp(self):
        self.num_clients = 10
        self.seed = 42
        self.num_samples = 2000
        rng = np.random.RandomState(self.seed)

        # Create multi-category attack metadata and binary targets
        # 1,200 Normal (0), 200 DoS (1), 200 Fuzzy (1), 200 Gear (1), 200 RPM (1)
        self.y_types_synth = np.array(
            ["Normal"] * 1200 + ["DoS"] * 200 + ["Fuzzy"] * 200 + ["Gear"] * 200 + ["RPM"] * 200,
            dtype=object,
        )
        self.y_synth = np.array([0 if t == "Normal" else 1 for t in self.y_types_synth], dtype=np.int64)

        perm = rng.permutation(self.num_samples)
        self.y_types_synth = self.y_types_synth[perm]
        self.y_synth = self.y_synth[perm]
        self.X_synth = rng.randn(self.num_samples, 16, 11).astype(np.float32)

    def test_dirichlet_partition_completeness_and_no_overlap(self):
        """Verify that Dirichlet partitioning assigns all samples with zero overlap or omission."""
        for alpha in [1.0, 0.5, 0.1]:
            partitions = partition_dirichlet(
                categories=self.y_types_synth,
                alpha=alpha,
                num_clients=self.num_clients,
                seed=self.seed,
                min_samples_per_client=30,
            )

            self.assertEqual(len(partitions), self.num_clients)
            all_indices = []
            for p in partitions:
                all_indices.extend(p.tolist())

            # Completeness: all samples assigned
            self.assertEqual(len(all_indices), self.num_samples)
            # Mutual exclusivity: zero duplicate indices
            self.assertEqual(len(set(all_indices)), self.num_samples)
            # Exactly matches index set {0, ..., num_samples - 1}
            self.assertEqual(set(all_indices), set(range(self.num_samples)))

    def test_dirichlet_deterministic_reproducibility(self):
        """Verify that identical seeds produce identical Dirichlet partitions."""
        part1 = partition_dirichlet(self.y_types_synth, alpha=0.5, num_clients=self.num_clients, seed=self.seed, min_samples_per_client=30)
        part2 = partition_dirichlet(self.y_types_synth, alpha=0.5, num_clients=self.num_clients, seed=self.seed, min_samples_per_client=30)

        for p1, p2 in zip(part1, part2):
            np.testing.assert_array_equal(p1, p2)

        part_diff = partition_dirichlet(self.y_types_synth, alpha=0.5, num_clients=self.num_clients, seed=999, min_samples_per_client=30)
        diff_found = any(not np.array_equal(p1, pd) for p1, pd in zip(part1, part_diff))
        self.assertTrue(diff_found, "Different seeds must produce different partitions.")

    def test_dirichlet_alpha_validation_and_errors(self):
        """Verify error handling for invalid alpha, invalid client count, or impossible constraints."""
        with self.assertRaises(ValueError):
            partition_dirichlet(self.y_types_synth, alpha=0.0)

        with self.assertRaises(ValueError):
            partition_dirichlet(self.y_types_synth, alpha=-0.5)

        with self.assertRaises(ValueError):
            partition_dirichlet(self.y_types_synth, alpha=1.0, num_clients=0)

        # Exceeds total samples: 10 clients * 300 min_samples = 3,000 > 2,000 samples
        with self.assertRaises(ValueError):
            partition_dirichlet(self.y_types_synth, alpha=1.0, num_clients=10, min_samples_per_client=300)

    def test_dirichlet_client_distribution_summaries(self):
        """Verify create_noniid_client_datasets returns well-formed diagnostics."""
        client_datasets, summary = create_noniid_client_datasets(
            X_train=self.X_synth,
            y_train=self.y_synth,
            y_types_train=self.y_types_synth,
            alpha=0.5,
            num_clients=self.num_clients,
            seed=self.seed,
            min_samples_per_client=30,
        )

        self.assertEqual(len(client_datasets), self.num_clients)
        self.assertEqual(summary["total_training_samples"], self.num_samples)
        self.assertEqual(summary["alpha"], 0.5)
        self.assertEqual(len(summary["clients"]), self.num_clients)

        total_collected = sum(c["num_samples"] for c in summary["clients"])
        self.assertEqual(total_collected, self.num_samples)

        # Check per-client attack types dictionary
        for c in summary["clients"]:
            self.assertIn("attack_types", c)
            sum_types = sum(c["attack_types"].values())
            self.assertEqual(sum_types, c["num_samples"])

    def test_dirichlet_binary_target_preservation(self):
        """Verify that local y arrays remain strictly binary (0 or 1)."""
        client_datasets, _ = create_noniid_client_datasets(
            X_train=self.X_synth,
            y_train=self.y_synth,
            y_types_train=self.y_types_synth,
            alpha=0.1,
            num_clients=self.num_clients,
            seed=self.seed,
            min_samples_per_client=20,
        )

        for cd in client_datasets:
            unique_y = np.unique(cd["y"])
            for val in unique_y:
                self.assertIn(val, [0, 1])

    def test_dirichlet_attack_type_alignment(self):
        """Verify that client y_types matches y (Normal -> 0, others -> 1)."""
        client_datasets, _ = create_noniid_client_datasets(
            X_train=self.X_synth,
            y_train=self.y_synth,
            y_types_train=self.y_types_synth,
            alpha=1.0,
            num_clients=self.num_clients,
            seed=self.seed,
            min_samples_per_client=30,
        )

        for cd in client_datasets:
            types = cd["y_types"]
            y = cd["y"]
            for t, target in zip(types, y):
                if t == "Normal":
                    self.assertEqual(target, 0)
                else:
                    self.assertEqual(target, 1)

    def test_dirichlet_small_categories_handling(self):
        """Verify partitioning handles small categories without crashing."""
        small_types = np.array(["CatA"] * 20 + ["CatB"] * 30 + ["CatC"] * 50)
        partitions = partition_dirichlet(small_types, alpha=1.0, num_clients=3, seed=self.seed, min_samples_per_client=10)
        self.assertEqual(len(partitions), 3)
        all_idxs = [idx for p in partitions for idx in p]
        self.assertEqual(len(all_idxs), 100)
        self.assertEqual(len(set(all_idxs)), 100)


if __name__ == "__main__":
    unittest.main()
