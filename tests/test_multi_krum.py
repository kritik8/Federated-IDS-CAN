"""
Unit and integration tests for Multi-Krum Byzantine-robust aggregation.
"""

import unittest
import copy
import numpy as np
import torch
from src.federated.aggregation import multi_krum_aggregate, aggregate_updates
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


class TestMultiKrum(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_multi_krum_output_shape_and_dtype(self):
        """Verify output tensor shapes and dtypes match client updates."""
        client_updates = [
            ({"weight": torch.randn(5, 5, dtype=torch.float32), "bias": torch.randn(5, dtype=torch.float32)}, 100)
            for _ in range(10)
        ]
        agg_state = multi_krum_aggregate(client_updates, num_byzantine=2)
        self.assertEqual(agg_state["weight"].shape, (5, 5))
        self.assertEqual(agg_state["bias"].shape, (5,))
        self.assertEqual(agg_state["weight"].dtype, torch.float32)
        self.assertEqual(agg_state["bias"].dtype, torch.float32)

    def test_multi_krum_identical_updates(self):
        """When all clients submit identical updates, Multi-Krum must return that exact update."""
        identical_weight = torch.full((4, 4), 3.14159, dtype=torch.float32)
        identical_bias = torch.full((4,), -0.5, dtype=torch.float32)
        client_updates = [
            ({"weight": identical_weight.clone(), "bias": identical_bias.clone()}, 50)
            for _ in range(10)
        ]
        agg_state = multi_krum_aggregate(client_updates, num_byzantine=2)
        torch.testing.assert_close(agg_state["weight"], identical_weight)
        torch.testing.assert_close(agg_state["bias"], identical_bias)

    def test_multi_krum_deterministic_ranking_and_selection(self):
        """
        Construct a deterministic 1D parameter test case:
        8 benign clients clustered tightly around 0.0, 2 Byzantine clients far away at 100.0.
        Multi-Krum (n=10, f=2, m=6) must select from the benign cluster and reject Byzantine clients.
        """
        # Benign clients: 0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.05
        benign_vals = [0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.05]
        # Byzantine clients (indices 8, 9): 100.0, -100.0
        byz_vals = [100.0, -100.0]

        all_vals = benign_vals + byz_vals
        client_updates = [
            ({"w": torch.tensor([val], dtype=torch.float32)}, 100)
            for val in all_vals
        ]

        agg_state, diag = multi_krum_aggregate(
            client_updates, num_byzantine=2, return_diagnostics=True
        )

        selected = diag["selected_client_indices"]
        self.assertEqual(len(selected), 6)  # m = n - f - 2 = 10 - 2 - 2 = 6
        # Neither client 8 nor client 9 should be selected
        self.assertNotIn(8, selected)
        self.assertNotIn(9, selected)

        # Selected updates must be solely from benign clients
        for idx in selected:
            self.assertIn(idx, range(8))

        # Output must be the arithmetic mean of the selected updates
        expected_mean = float(np.mean([all_vals[idx] for idx in selected]))
        self.assertAlmostEqual(agg_state["w"].item(), expected_mean, places=5)

    def test_multi_krum_custom_m(self):
        """Verify custom num_selected m works properly."""
        client_updates = [
            ({"w": torch.tensor([float(i)], dtype=torch.float32)}, 10)
            for i in range(10)
        ]
        # If m = 1, standard Krum selects exactly 1 update
        agg_state, diag = multi_krum_aggregate(
            client_updates, num_byzantine=2, num_selected=1, return_diagnostics=True
        )
        self.assertEqual(len(diag["selected_client_indices"]), 1)

    def test_multi_krum_feasibility_condition_n_ge_2f_plus_3(self):
        """Multi-Krum requires n >= 2f + 3. Check that violation raises ValueError."""
        # For f=2, min n is 7. If n=6, should raise ValueError.
        client_updates_6 = [
            ({"w": torch.tensor([1.0], dtype=torch.float32)}, 10)
            for _ in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            multi_krum_aggregate(client_updates_6, num_byzantine=2)
        self.assertIn("Multi-Krum requires n >= 2f + 3", str(ctx.exception))

        # Negative f should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            multi_krum_aggregate(client_updates_6, num_byzantine=-1)
        self.assertIn("num_byzantine must be non-negative", str(ctx.exception))

    def test_multi_krum_no_input_modification(self):
        """Ensure input updates are not mutated in place."""
        orig_tensors = [torch.randn(3, 3) for _ in range(10)]
        input_tensors = [t.clone() for t in orig_tensors]
        client_updates = [({"w": t}, 10) for t in input_tensors]

        _ = multi_krum_aggregate(client_updates, num_byzantine=2)

        for orig, current in zip(orig_tensors, input_tensors):
            torch.testing.assert_close(orig, current)

    def test_multi_krum_dispatcher_integration(self):
        """Test dispatcher via aggregate_updates()."""
        client_updates = [
            ({"w": torch.tensor([float(i)], dtype=torch.float32)}, 10)
            for i in range(10)
        ]
        res = aggregate_updates(client_updates, method="multi_krum", num_byzantine=2)
        self.assertIsInstance(res, dict)
        self.assertIn("w", res)

    def test_end_to_end_server_with_multi_krum(self):
        """Integration test with FederatedServer over 1 round of training."""
        X_synth = np.random.randn(100, 16, 11).astype(np.float32)
        y_synth = np.array([0] * 50 + [1] * 50, dtype=np.int64)

        # Need at least n=7 clients for f=2 (let's use n=10)
        clients = [
            FederatedClient(client_id=i, X=X_synth[i*10:(i+1)*10], y=y_synth[i*10:(i+1)*10], batch_size=32)
            for i in range(10)
        ]
        server = FederatedServer(
            val_data=(X_synth[:20], y_synth[:20]),
            seed=42,
            aggregation_method="multi_krum",
            aggregation_kwargs={"num_byzantine": 2, "return_diagnostics": True},
        )
        server.register_clients(clients)

        res = server.fit(num_rounds=1, local_epochs=1, verbose=False)
        self.assertEqual(res["best_round"], 1)
        self.assertIn("val_f1", server.history)
        self.assertIn("aggregation_diagnostics", server.history)
        diag_round_1 = server.history["aggregation_diagnostics"][0]
        self.assertEqual(diag_round_1["m_selected"], 6)
        self.assertEqual(len(diag_round_1["selected_client_indices"]), 6)


if __name__ == "__main__":
    unittest.main()

