"""
Regression and Invariant Tests for Final Research Freeze (Iteration 3).
"""

import json
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_confusion_matrix_collapse_invariant():
    """Verify exact theoretical metrics when a model collapses to predicting all-positive."""
    tn, fp, fn, tp = 0, 2900, 0, 1790
    total = tn + fp + fn + tp
    assert total == 4690
    acc = (tp + tn) / total
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    assert abs(acc - (1790 / 4690)) < 1e-7
    assert abs(prec - (1790 / 4690)) < 1e-7
    assert rec == 1.0
    assert abs(f1 - (179 / 324)) < 1e-7
    assert fpr == 1.0
    assert fnr == 0.0

def test_summary_standard_deviation_ddof1():
    """Verify that robust aggregators use sample standard deviation (ddof=1)."""
    summary_path = PROJECT_ROOT / "results" / "metrics" / "robust_all_aggregators_summary.json"
    assert summary_path.exists()

    with open(summary_path, "r") as f:
        summary = json.load(f)

    with open(PROJECT_ROOT / "results" / "metrics" / "robust_multiseed_summary.json", "r") as f:
        rms = json.load(f)

    for k, v in rms.items():
        runs = v["runs"]
        f1s = [r["test_f1"] for r in runs]
        expected_std = float(np.std(f1s, ddof=1))
        assert abs(v["f1_std"] - expected_std) < 1e-5, f"{k} f1_std must match ddof=1 sample std"

def test_random_forest_zero_fpr_property():
    """Verify that Random Forest produced zero false positives across all 3 seeds."""
    rf_path = PROJECT_ROOT / "results" / "metrics" / "baseline_random_forest.json"
    assert rf_path.exists()

    with open(rf_path, "r") as f:
        rf_data = json.load(f)

    for seed_str, run_data in rf_data.items():
        m = run_data["test_metrics"]
        assert m["false_positives"] == 0, f"Seed {seed_str} RF had non-zero false positives"
        assert m["false_positive_rate"] == 0.0
        assert m["precision"] == 1.0

def test_final_verified_results_structure():
    """Verify that final_verified_results.json exists and contains 18 certified records."""
    verified_path = PROJECT_ROOT / "results" / "metrics" / "final_verified_results.json"
    assert verified_path.exists()

    with open(verified_path, "r") as f:
        records = json.load(f)

    assert len(records) == 18, f"Expected 18 records, found {len(records)}"
    categories = {r["Category"] for r in records}
    assert "Centralized Baseline" in categories
    assert "Federated IID" in categories
    assert "Federated Non-IID" in categories
    assert "Byzantine-Robust FL" in categories

def test_authoritative_csv_and_checks_exist():
    """Verify all required Iteration 3 deliverable artifacts exist."""
    assert (PROJECT_ROOT / "results" / "metrics" / "final_verified_results.csv").exists()
    assert (PROJECT_ROOT / "results" / "metrics" / "final_validation_checks.json").exists()
    assert (PROJECT_ROOT / "results" / "metrics" / "latency_and_size_audit.json").exists()
    assert (PROJECT_ROOT / "final_audit_2026" / "ITERATION_3_AUDIT_FINDINGS.md").exists()
    assert (PROJECT_ROOT / "final_audit_2026" / "NONIID_RESULT_RECONCILIATION.md").exists()
    assert (PROJECT_ROOT / "final_audit_2026" / "LATENCY_SPEEDUP_RECONCILIATION.md").exists()

def test_noniid_reconciliation_invariants():
    """Verify that Non-IID Dirichlet summary metrics strictly adhere to the reconciled authoritative values."""
    with open(PROJECT_ROOT / "results" / "multiseed_summary_noniid.json") as f:
        d = json.load(f)

    # Alpha 0.1
    assert abs(d["0.1"]["mean"]["F1_Score"] - 0.69159165) < 1e-5
    assert abs(d["0.1"]["std"]["F1_Score"] - 0.17864558) < 1e-5

    # Alpha 0.5
    assert abs(d["0.5"]["mean"]["F1_Score"] - 0.97407432) < 1e-5
    assert abs(d["0.5"]["std"]["F1_Score"] - 0.00504553) < 1e-5

    # Alpha 1.0
    assert abs(d["1.0"]["mean"]["F1_Score"] - 0.97574409) < 1e-5
    assert abs(d["1.0"]["std"]["F1_Score"] - 0.00814278) < 1e-5

def test_latency_speedup_ratios():
    """Verify that model-only latency speedups strictly equal 66.8x (single) and 157.7x (multi)."""
    with open(PROJECT_ROOT / "results" / "metrics" / "latency_and_size_audit.json") as f:
        audit = json.load(f)

    speedups = audit["calculated_speedups_batch_1"]
    cnn_lat = speedups["cnn_baseline_latency_us"]
    assert abs(cnn_lat - 455.99) < 1e-2

    st_speedup = speedups["rf_single_thread_model_only_speedup"]
    mt_speedup = speedups["rf_multi_thread_model_only_speedup"]
    assert abs(st_speedup - 66.8) < 1e-1
    assert abs(mt_speedup - 157.7) < 1e-1


