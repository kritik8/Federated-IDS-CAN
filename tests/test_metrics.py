"""
Unit tests for Performance Evaluation and Profiling Module (src.evaluation.metrics)
"""

import unittest
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from src.evaluation.metrics import compute_classification_metrics


class TestMetrics(unittest.TestCase):
    def test_single_class_roc_auc_returns_none_without_warning(self):
        """Single-class targets must return None for roc_auc with zero warnings."""
        y_true = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)
        y_prob = np.random.uniform(0.1, 0.4, size=50)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            metrics = compute_classification_metrics(y_true, y_pred, y_prob)

            # Verify no warnings were emitted
            undefined_warnings = [w for w in recorded_warnings if issubclass(w.category, UndefinedMetricWarning)]
            self.assertEqual(len(undefined_warnings), 0)

        self.assertIn("roc_auc", metrics)
        self.assertIsNone(metrics["roc_auc"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_valid_two_class_roc_auc_returns_float(self):
        """Valid binary targets must produce a valid float roc_auc in [0.0, 1.0]."""
        y_true = np.array([0] * 30 + [1] * 20, dtype=int)
        y_pred = np.array([0] * 28 + [1] * 2 + [0] * 1 + [1] * 19, dtype=int)
        # Probabilities correlate strongly with y_true
        y_prob = np.concatenate([
            np.random.uniform(0.0, 0.3, size=30),
            np.random.uniform(0.7, 1.0, size=20)
        ])

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsInstance(metrics["roc_auc"], float)
        self.assertGreaterEqual(metrics["roc_auc"], 0.9)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

    def test_metric_correctness(self):
        """Verify accuracy, precision, recall, f1, fpr, fnr, and confusion matrix."""
        # Known ground truth and predictions:
        # TN=15, FP=5, FN=3, TP=17
        y_true = np.array([0]*20 + [1]*20)
        y_pred = np.array([0]*15 + [1]*5 + [0]*3 + [1]*17)

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertEqual(metrics["true_negatives"], 15)
        self.assertEqual(metrics["false_positives"], 5)
        self.assertEqual(metrics["false_negatives"], 3)
        self.assertEqual(metrics["true_positives"], 17)

        self.assertAlmostEqual(metrics["accuracy"], 32.0 / 40.0)
        self.assertAlmostEqual(metrics["precision"], 17.0 / 22.0)
        self.assertAlmostEqual(metrics["recall"], 17.0 / 20.0)
        self.assertAlmostEqual(metrics["false_positive_rate"], 5.0 / 20.0)
        self.assertAlmostEqual(metrics["false_negative_rate"], 3.0 / 20.0)

        f1_expected = 2 * (17/22) * (17/20) / ((17/22) + (17/20))
        self.assertAlmostEqual(metrics["f1_score"], f1_expected)
        self.assertEqual(metrics["confusion_matrix"], [[15, 5], [3, 17]])
        self.assertNotIn("roc_auc", metrics)  # Since y_prob was None


if __name__ == "__main__":
    unittest.main()

