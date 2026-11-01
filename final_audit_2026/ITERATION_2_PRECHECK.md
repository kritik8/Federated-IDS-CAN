# Iteration 2 Pre-Check & Iteration 1 Validation Gate

**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Date:** October 31, 2026  
**Auditor / Implementation Lead:** Senior Research Systems Engineer  
**Document Path:** `final_audit_2026/ITERATION_2_PRECHECK.md`

---

## 1. Executive Gate Certification

Before embarking on Iteration 2 (P1) tasks (Multi-Krum aggregation and Centralized Random Forest baseline), a formal validation gate was conducted across all Iteration 1 outputs and configurations.

**Validation Status:** **PASSED WITHOUT BLOCKERS (GREEN)**  
All historical artifacts remain intact, multi-seed results across all regimes are verified from raw JSON/CSV data, and unit tests pass with zero warnings.

---

## 2. Artifacts Inspected & Verified

| Category | File Path | Validation Result | Notes |
|---|---|:---:|---|
| **Iteration 1 Report** | `final_audit_2026/ITERATION_1_REPORT.md` | **VERIFIED** | Reports match raw experiment JSON files. |
| **Baseline Sensitivity** | `final_audit_2026/BASELINE_SEED_ANALYSIS.md` | **VERIFIED** | Analysis matches independent recalculation. |
| **Project README** | `README.md` | **VERIFIED** | Malicious clients [8, 9], $\gamma=1.0$, data weights documented. |
| **Centralized Baseline** | `results/baseline_multiseed/seed_{42,43,44}/metrics.json` | **VERIFIED** | Seed 42 matches historical `baseline_results.json` exactly. |
| **IID Baseline** | `results/iid_multiseed/seed_{42,43,44}/metrics.json` | **VERIFIED** | Seed 42 matches historical `federated_iid_fedavg.json` exactly. |
| **Non-IID Baselines** | `results/noniid_multiseed/alpha_{1.0,0.5,0.1}/seed_{42,43,44}/metrics.json` | **VERIFIED** | Seed 42 matches historical `federated_noniid_alpha_{a}.json` exactly. |
| **Summary Files** | `results/multiseed_summary_*.{json,csv}` | **VERIFIED** | Recalculated with sample std ($\text{ddof}=1$), matches to $<10^{-5}$. |
| **Attack B Figure** | `results/figures/robust_per_attack_recall_attack_b.png` | **QUALIFIED** | Updated title & notes to specify Seed 42 benchmark context. |
| **Architecture Diagram** | `results/figures/system_architecture.png` | **QUALIFIED** | Removed any ambiguous fleet terminology; explicitly states 10 simulated partitions on Kia Soul stream. |
| **Evaluation Metrics** | `src/evaluation/metrics.py` | **HARDENED** | Narrowed exception to `ValueError`; verified ROC-AUC edge cases. |
| **Test Suite** | `tests/test_metrics.py`, `tests/` | **PASSED** | 38 passed, 0 failed, 0 warnings in 17.06s. |

---

## 3. Discrepancies Discovered & Corrections Made

1. **Figure 1 (Architecture Diagram) Fleet Representation:**
   - *Discovery:* Visual labels in `results/figures/system_architecture.png` referred to "Heterogeneous Vehicular Fleet" and "ECUs / Vehicles", which could be misinterpreted as multiple physical test vehicles.
   - *Correction:* Updated script `scratch/plot_architecture.py` to explicitly specify "Simulated Vehicular Fleet (K=10 Virtual Partitions from Kia Soul CAN Stream)" and added explicit qualification in the caption. Re-rendered at 300 DPI.
2. **Attack B Per-Attack Recall Single-Seed Clarification:**
   - *Discovery:* `results/figures/robust_per_attack_recall_attack_b.png` displayed Attack B per-class recall without an explicit benchmark seed indication in the title.
   - *Correction:* Updated `scratch/plot_attack_b_recall.py` to explicitly note `[Primary Matched Benchmark: Seed 42, Non-IID Dirichlet α=0.5, 10 Clients]` and noted test partition details. Re-rendered at 300 DPI.
3. **Metric Exception Scope & Test Coverage:**
   - *Discovery:* `src/evaluation/metrics.py` used `except Exception:` when computing ROC-AUC, and there was no dedicated unit test file for `compute_classification_metrics`.
   - *Correction:* Scoped exception to `except ValueError:` and added comprehensive unit tests in `tests/test_metrics.py` testing single-class ROC-AUC (returns `None` without warnings), valid binary ROC-AUC, and full metric correctness.

---

## 4. Threat Model Final Confirmation

- Number of simulated clients: $n = 10$.
- Designated Byzantine clients: `client_8` and `client_9` ($f=2$, 20% by client count).
- Data weight distribution under Dirichlet $\alpha = 0.5$ (21,875 total training windows):
  - **Seed 42:** `client_8` = 3,572, `client_9` = 5,101 $\implies$ **39.65%** of training samples.
  - **Seed 43:** `client_8` = 844, `client_9` = 4,623 $\implies$ **24.99%** of training samples.
  - **Seed 44:** `client_8` = 2,983, `client_9` = 5,383 $\implies$ **38.24%** of training samples.
- Model update sign reversal scaling parameter: $\gamma = 1.0$.

---

## 5. Formal Gate Verdict

Iteration 1 artifacts are certified as verified, reproducible, and mathematically consistent. Authorization is granted to proceed with Iteration 2 (P1) tasks:
- **Phase 1 & 2:** Multi-Krum implementation, unit tests, and multi-seed Byzantine robustness evaluation.
- **Phase 3:** Centralized Random Forest baseline on flattened $(N, 176)$ windows.

