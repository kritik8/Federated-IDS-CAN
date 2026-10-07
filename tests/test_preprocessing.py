"""
Unit tests for preprocessing, leak-free delta-t computation, and feature scaling.
"""

import unittest
import numpy as np
import pandas as pd

from src.preprocessing.parser import parse_hex_id, parse_hex_payload, compute_inter_arrival_times
from src.preprocessing.pipeline import CANPreprocessor, create_sliding_windows


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing parser, normalization, and window creation."""

    def test_leak_free_inter_arrival_times(self):
        """Verify index 0 is 0.0 and does not leak future timestamps."""
        ts = np.array([100.0, 100.005, 100.015, 100.030, 200.0])
        deltas = compute_inter_arrival_times(ts, max_delta=1.0, normalize_log=False)

        # Index 0 must be exactly 0.0 (leak-free)
        self.assertEqual(deltas[0], 0.0)
        self.assertAlmostEqual(deltas[1], 0.005, places=5)
        self.assertAlmostEqual(deltas[2], 0.010, places=5)
        self.assertAlmostEqual(deltas[3], 0.015, places=5)
        self.assertAlmostEqual(deltas[4], 1.0, places=5)  # Clipped to max_delta

    def test_delta_t_domain_scaling_unit_interval(self):
        """Verify delta-t values are strictly bounded in [0.0, 1.0]."""
        ts = np.array([0.0, 0.0001, 0.005, 0.05, 1.0, 10.0])
        deltas = compute_inter_arrival_times(
            ts,
            max_delta=1.0,
            normalize_log=True,
            scale_to_unit_interval=True,
        )

        self.assertTrue(np.all(deltas >= 0.0))
        self.assertTrue(np.all(deltas <= 1.0))
        # At delta = 0.0, normalized log value must be 0.0
        self.assertAlmostEqual(deltas[0], 0.0, places=5)
        # At maximum delta (1.0s, clipped), normalized value must be 1.0
        self.assertAlmostEqual(deltas[-1], 1.0, places=5)

    def test_can_preprocessor_full_pipeline(self):
        """Verify CANPreprocessor outputs 11 features, all in [0.0, 1.0]."""
        df = pd.DataFrame({
            "Timestamp": [10.0, 10.002, 10.004, 10.007],
            "CAN_ID": ["0316", "02b0", "0000", "07ff"],
            "DLC": [8, 5, 8, 2],
            "DATA[0]": ["05", "ff", "00", "01"],
            "DATA[1]": ["28", "7f", "00", "00"],
            "DATA[2]": ["84", "00", "00", "00"],
            "DATA[3]": ["66", "05", "00", "00"],
            "DATA[4]": ["6d", "49", "00", "00"],
            "DATA[5]": ["00", "00", "00", "00"],
            "DATA[6]": ["00", "00", "00", "00"],
            "DATA[7]": ["a2", "00", "00", "00"],
            "Flag": ["R", "R", "T", "R"],
            "Attack_Type": ["Normal", "Normal", "DoS", "Normal"],
        })

        preprocessor = CANPreprocessor()
        feats, b_labels, a_types = preprocessor.transform_dataframe(df)

        self.assertEqual(feats.shape, (4, 11))
        self.assertTrue(np.all(feats >= 0.0))
        self.assertTrue(np.all(feats <= 1.0))
        np.testing.assert_array_equal(b_labels, [0, 0, 1, 0])
        np.testing.assert_array_equal(a_types, ["Normal", "Normal", "DoS", "Normal"])


if __name__ == "__main__":
    unittest.main()
