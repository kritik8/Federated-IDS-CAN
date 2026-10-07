"""
Unit tests for 1D-CNN baseline model architecture with GroupNorm.
"""

import unittest
import torch
import torch.nn as nn

from src.models.cnn1d import CAN1DCNN, count_parameters, get_group_norm


class TestCAN1DCNN(unittest.TestCase):
    """Test suite for CAN1DCNN architecture, GroupNorm layers, and forward passes."""

    def test_group_norm_layers(self):
        """Verify that GroupNorm is used instead of BatchNorm1d."""
        model = CAN1DCNN(in_channels=11, seq_length=16, num_classes=2)
        
        # Check normalization layers
        self.assertIsInstance(model.gn1, nn.GroupNorm)
        self.assertIsInstance(model.gn2, nn.GroupNorm)
        self.assertIsInstance(model.gn3, nn.GroupNorm)

        # Check group counts
        self.assertEqual(model.gn1.num_groups, 4)
        self.assertEqual(model.gn2.num_groups, 4)
        self.assertEqual(model.gn3.num_groups, 4)

        # Ensure no BatchNorm layers exist
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
        self.assertFalse(has_bn, "Model must not contain any BatchNorm layers.")

        # Ensure no running mean/var buffers exist
        buffer_names = [name for name, _ in model.named_buffers()]
        self.assertEqual(len(buffer_names), 0, f"Expected 0 non-trainable buffers, got {buffer_names}")

    def test_parameter_count(self):
        """Verify trainable parameters count matches expected 40,610."""
        model = CAN1DCNN(in_channels=11, seq_length=16, num_classes=2)
        params = count_parameters(model)
        self.assertEqual(params["trainable_parameters"], 40610)
        self.assertEqual(params["total_parameters"], 40610)

    def test_forward_pass_both_tensor_layouts(self):
        """Verify model handles (B, Seq_Len, Feats) and (B, Feats, Seq_Len)."""
        model = CAN1DCNN(in_channels=11, seq_length=16, num_classes=2)
        model.eval()

        # Layout 1: (Batch=8, Seq_Len=16, Feats=11)
        x1 = torch.randn(8, 16, 11)
        out1 = model(x1)
        self.assertEqual(out1.shape, (8, 2))

        # Layout 2: (Batch=8, Feats=11, Seq_Len=16)
        x2 = torch.randn(8, 11, 16)
        out2 = model(x2)
        self.assertEqual(out2.shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
