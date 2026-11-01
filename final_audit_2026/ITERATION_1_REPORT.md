# Iteration 1 (P0 Tasks) Execution & Completion Report

**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Target Venue:** Academic Conference / Journal Submission  
**Implementation Freeze:** November 8, 2026 | **Submission Deadline:** November 12, 2026  
**Date of Execution:** October 31, 2026  
**Auditor / Implementation Lead:** Independent Senior Research Systems Engineer & Machine Learning Specialist  
**Deliverable Path:** `final_audit_2026/ITERATION_1_REPORT.md`

---

## 1. Executive Summary & Deliverables Scorecard

Iteration 1 focused strictly on **Priority 0 (P0) critical tasks** identified during the independent technical and scientific audit. All tasks have been executed, experimentally verified, and integrated into the repository without modifying historical Seed 42 benchmark artifacts or altering the fundamental research methodology.

| Task ID | Component / Objective | Target Artifact | Status | Verification Detail |
|---|---|---|:---:|---|
| **TASK-P0-1** | Fix README Threat Model Discrepancy | `README.md` | **COMPLETED** | Corrected malicious client IDs from `[0, 1]` to `[8, 9]`; corrected $\gamma$ from $1.5$ to $1.0$; documented variable sample weights across seeds (24.99% – 39.65%). |
| **TASK-P0-2A** | Multi-Seed IID Baseline Experiments | `results/iid_multiseed/` | **COMPLETED** | Executed Seeds 43, 44 under identical 10-client, 10-round FedAvg setting; Seed 42 preserved. Mean F1 = $96.94\% \pm 1.19\%$. |
| **TASK-P0-2B** | Multi-Seed Non-IID Dirichlet Experiments | `results/noniid_multiseed/` | **COMPLETED** | Executed Seeds 43, 44 across all three Dirichlet regimes ($\alpha \in \{1.0, 0.5, 0.1\}$); Seed 42 preserved. Multi-seed tables generated. |
| **TASK-P0-2C** | Centralized Baseline Seed Sensitivity | `results/baseline_multiseed/` | **COMPLETED** | Executed Seeds 43, 44 of Centralized 1D-CNN (GroupNorm); Seed 42 preserved. Mean F1 = $99.54\% \pm 0.13\%$, FPR = $0.22\% \pm 0.07\%$. |
| **TASK-P0-3** | Attack B Companion Recall Visualization | `results/figures/robust_per_attack_recall_attack_b.png` | **COMPLETED** | Generated publication-quality 300 DPI bar chart displaying per-class recall (DoS, Fuzzy, Gear, RPM, Normal) under Byzantine model update corruption. |
| **TASK-P0-4** | End-to-End System Architecture Diagram | `results/figures/system_architecture.png` | **COMPLETED** | Rendered publication-quality 300 DPI diagram covering in-vehicle telemetry, 11D windowing, GroupNorm CNN, fleet Dirichlet skew, and Byzantine aggregation. |
| **TASK-P0-5** | Test Suite Warning Resolution | `src/evaluation/metrics.py` | **COMPLETED** | Fixed `UndefinedMetricWarning` by verifying class diversity before invoking `roc_auc_score`. Test suite: **35 passed, 0 failed, 0 warnings** in 21.03s. |
| **TASK-P0-6** | Multi-Seed Summary Aggregations | `results/multiseed_summary_*.{json,csv}` | **COMPLETED** | Computed matched tables reporting Mean $\pm$ Sample Std ($N=3$, $\text{ddof}=1$) across all regimes. |

---

## 2. Phase 1: Threat Model Inconsistency Resolution (`README.md`)

### Discrepancy Identified:
- **Historical README Claim:** Designated malicious clients as `client_0` and `client_1` with scaling factor $\gamma = 1.5$.
- **Actual Experimental Ground Truth:** Configured in `configs/robust_federated_config.json` and executed in `results/metrics/robust_federated_experiments.json` with malicious clients `[8, 9]` and scaling factor $\gamma = 1.0$.

### Resolution Implemented:
`README.md` (lines 145–165) was updated to accurately specify:
1. **Adversary Budget:** $f=2$ (20% of fleet by client count), specifically designated as `client_8` and `client_9`.
2. **Quantity Skew Documentation:** Explicitly documented that equal client count does not imply equal training sample weight under Dirichlet non-IID distributions ($\alpha = 0.5$):
   - **Seed 42:** `client_8` = 3,572, `client_9` = 5,101 $\implies$ 8,673 samples (**39.65%** of training data).
   - **Seed 43:** `client_8` = 844, `client_9` = 4,623 $\implies$ 5,467 samples (**24.99%** of training data).
   - **Seed 44:** `client_8` = 2,983, `client_9` = 5,383 $\implies$ 8,366 samples (**38.24%** of training data).
3. **Attack Formulations:**
   - **Attack A (Label Flipping):** $y \mapsto 1 - y$ for all local binary training labels ($0 \to 1$ and $1 \to 0$).
   - **Attack B (Model-Update Sign Reversal):** $\Delta_k' = -\gamma \cdot \Delta_k$ where $\Delta_k = \theta_k^{(t)} - \theta^{(t-1)}$. With benchmark configuration $\gamma = 1.0$, the transmitted poisoned model state is $\theta_k' = \theta^{(t-1)} - (\theta_k^{(t)} - \theta^{(t-1)}) = 2\theta^{(t-1)} - \theta_k^{(t)}$.

---

## 3. Phase 2 & 3: Complete Multi-Seed Experimental Results

All experiments evaluated exactly the same held-out test partition (**4,690 CAN message windows:** 2,900 Normal, 1,790 Cyber-Attacks) across Seeds 42, 43, and 44. All sample standard deviations are computed with Bessel's correction ($\text{ddof}=1$).

### Master Comprehensive Results Table

| Experimental Regime | Seed 42 | Seed 43 | Seed 44 | Mean ± Std ($N=3$, ddof=1) |
|---|:---:|:---:|:---:|:---:|
| **Centralized Baseline (1D-CNN)** | | | | |
| - Best Iteration | Ep 8 | Ep 10 | Ep 10 | — |
| - Best Validation F1 (%) | 99.42% | 99.55% | 99.67% | **99.55% ± 0.13%** |
| - Test Accuracy (%) | 99.66% | 99.55% | 99.74% | **99.65% ± 0.10%** |
| - Test Precision (%) | 99.61% | 99.55% | 99.78% | **99.65% ± 0.12%** |
| - Test Recall (%) | 99.50% | 99.27% | 99.55% | **99.44% ± 0.15%** |
| - Test F1-Score (%) | 99.55% | 99.41% | 99.66% | **99.54% ± 0.13%** |
| - False Positive Rate (FPR %) | 0.24% | 0.28% | 0.14% | **0.22% ± 0.07%** |
| - False Negative Rate (FNR %) | 0.50% | 0.73% | 0.45% | **0.56% ± 0.15%** |
| - ROC-AUC | 0.9992 | 0.9996 | 0.9993 | **0.9994 ± 0.0002** |
| **IID Federated Baseline (FedAvg)** | | | | |
| - Best Iteration | Rd 9 | Rd 10 | Rd 9 | — |
| - Best Validation F1 (%) | 97.99% | 98.11% | 97.68% | **97.93% ± 0.22%** |
| - Test Accuracy (%) | 98.25% | 98.19% | 96.72% | **97.72% ± 0.87%** |
| - Test Precision (%) | 99.71% | 98.74% | 98.69% | **99.05% ± 0.57%** |
| - Test Recall (%) | 95.70% | 96.48% | 92.63% | **94.93% ± 2.04%** |
| - Test F1-Score (%) | 97.66% | 97.60% | 95.56% | **96.94% ± 1.19%** |
| - False Positive Rate (FPR %) | 0.17% | 0.76% | 0.76% | **0.56% ± 0.34%** |
| - False Negative Rate (FNR %) | 4.30% | 3.52% | 7.37% | **5.07% ± 2.04%** |
| - ROC-AUC | 0.9965 | 0.9964 | 0.9954 | **0.9961 ± 0.0006** |
| **Non-IID FedAvg ($\alpha = 1.0$)** | | | | |
| - Best Iteration | Rd 10 | Rd 10 | Rd 10 | — |
| - Best Validation F1 (%) | 98.05% | 98.57% | 97.68% | **98.10% ± 0.45%** |
| - Test Accuracy (%) | 98.51% | 98.55% | 97.51% | **98.19% ± 0.59%** |
| - Test Precision (%) | 99.83% | 99.20% | 99.59% | **99.54% ± 0.32%** |
| - Test Recall (%) | 96.26% | 96.98% | 93.85% | **95.70% ± 1.64%** |
| - Test F1-Score (%) | 98.01% | 98.08% | 96.64% | **97.57% ± 0.81%** |
| - False Positive Rate (FPR %) | 0.10% | 0.48% | 0.24% | **0.28% ± 0.19%** |
| - False Negative Rate (FNR %) | 3.74% | 3.02% | 6.15% | **4.30% ± 1.64%** |
| - ROC-AUC | 0.9961 | 0.9971 | 0.9959 | **0.9964 ± 0.0007** |
| **Non-IID FedAvg ($\alpha = 0.5$)** | | | | |
| - Best Iteration | Rd 9 | Rd 10 | Rd 10 | — |
| - Best Validation F1 (%) | 97.94% | 96.64% | 98.17% | **97.58% ± 0.82%** |
| - Test Accuracy (%) | 98.49% | 97.91% | 97.78% | **98.06% ± 0.37%** |
| - Test Precision (%) | 99.60% | 99.94% | 98.56% | **99.37% ± 0.72%** |
| - Test Recall (%) | 96.42% | 94.58% | 95.59% | **95.53% ± 0.92%** |
| - Test F1-Score (%) | 97.98% | 97.19% | 97.05% | **97.41% ± 0.50%** |
| - False Positive Rate (FPR %) | 0.24% | 0.03% | 0.86% | **0.38% ± 0.43%** |
| - False Negative Rate (FNR %) | 3.58% | 5.42% | 4.41% | **4.47% ± 0.92%** |
| - ROC-AUC | 0.9956 | 0.9975 | 0.9958 | **0.9963 ± 0.0010** |
| **Non-IID FedAvg ($\alpha = 0.1$)** | | | | |
| - Best Iteration | Rd 10 | Rd 9 | Rd 10 | — |
| - Best Validation F1 (%) | 90.12% | 55.54% | 65.43% | **70.37% ± 17.81%** |
| - Test Accuracy (%) | 92.11% | 75.76% | 68.25% | **78.71% ± 12.20%** |
| - Test Precision (%) | 97.91% | 99.25% | 56.07% | **84.41% ± 24.55%** |
| - Test Recall (%) | 81.06% | 36.76% | 77.71% | **65.18% ± 24.67%** |
| - Test F1-Score (%) | 88.69% | 53.65% | 65.14% | **69.16% ± 17.86%** |
| - False Positive Rate (FPR %) | 1.07% | 0.17% | 37.59% | **12.94% ± 21.35%** |
| - False Negative Rate (FNR %) | 18.94% | 63.24% | 22.29% | **34.82% ± 24.67%** |
| - ROC-AUC | 0.9573 | 0.8766 | 0.7988 | **0.8776 ± 0.0792** |

---

## 4. Phase 4: Generated Publication Figures

Two publication-quality figures were designed, generated, and verified at **300 DPI**:

1. **`results/figures/robust_per_attack_recall_attack_b.png` (199.95 KB):**
   - **Purpose:** Companion visualization to `robust_per_attack_recall.png` (which only plotted Attack A).
   - **Content:** Compares per-class intrusion detection accuracy across **DoS**, **Fuzzy**, **Gear**, **RPM**, and **Normal** traffic under Attack B (Model-Update Sign Reversal).
   - **Crucial Scientific Insight Illustrated:** FedAvg completely collapses on Normal traffic ($0.0\%$ specificity, producing 100% false alarms on legitimate packets), while Coordinate-wise Median ($98.3\%$ Normal accuracy, $97.7\%$ DoS, $97.5\%$ Gear, $98.8\%$ RPM) and Trimmed Mean ($97.5\%$ Normal accuracy) successfully isolate the Byzantine perturbation.

2. **`results/figures/system_architecture.png` (859.74 KB):**
   - **Purpose:** Comprehensive framework overview for Figure 1 of the manuscript.
   - **Content:** Three integrated architectural columns:
     - **In-Vehicle Edge Sensing:** Automotive CAN 2.0B differential bus $\to$ temporal window parser ($W=16, S=16$) $\to$ 11D feature representation $\to$ lightweight `CAN1DCNN` (GroupNorm, 40,610 parameters, 456 $\mu$s latency).
     - **Heterogeneous Vehicular Fleet:** 10 simulated vehicular ECUs partitioned under category-aware Dirichlet skew ($\alpha \in \{1.0, 0.5, 0.1\}$), with adversarial budget $f=2$ (Clients 8 & 9 executing Label Flipping and Model-Update Sign Reversal).
     - **Robust Central Orchestration:** Uplink/downlink communication ($3.10$ MB/round) with robust aggregation rules (Coordinate-wise Median, Coordinate-wise Trimmed Mean) mitigating gradient corruption.

---

## 5. Phase 6: Test Suite Warning Eradication

### Root Cause Analysis:
During `tests/test_robust_aggregation.py::TestThreatModel::test_end_to_end_server_with_robust_aggregation`, a synthetic validation set with 20 samples was created with only class `0` present (`y_synth[:20] = [0] * 20`). Invoking `sklearn.metrics.roc_auc_score(y_true, y_prob)` on single-class targets triggered three consecutive `UndefinedMetricWarning` warnings.

### Fix Implemented:
In `src/evaluation/metrics.py` (lines 71–79), an explicit class diversity verification was added before invoking `roc_auc_score`:
```python
if y_prob is not None:
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
```

### Verification:
```bash
python -m pytest tests
============================= 35 passed in 21.03s =============================
```
**Zero warnings emitted.** All 35 tests pass deterministically.

---

## 6. Scientific Analysis & Paper Discussion Points

1. **Centralized Ceiling Stability:**
   The centralized baseline is exceptionally stable ($99.54\% \pm 0.13\%$ F1). This establishes a solid empirical ceiling. Reviewers cannot argue that differences between centralized and federated models stem from stochastic training instability.

2. **IID vs. Moderate Non-IID Resilience:**
   - IID FedAvg achieves $96.94\% \pm 1.19\%$ F1.
   - Non-IID FedAvg ($\alpha = 1.0$) achieves $97.57\% \pm 0.81\%$ F1.
   - Non-IID FedAvg ($\alpha = 0.5$) achieves $97.41\% \pm 0.50\%$ F1.
   Under moderate heterogeneity ($\alpha \ge 0.5$), standard FedAvg remains remarkably robust, experiencing less than a $2.5\%$ drop compared to the centralized ceiling.

3. **Catastrophic Client Drift under Extreme Non-IID ($\alpha = 0.1$):**
   Under extreme Dirichlet concentration ($\alpha = 0.1$), performance severely degrades ($70.37\% \pm 17.81\%$ F1, with Seed 43 dropping to $53.65\%$ F1 and $63.24\%$ FNR). Because individual clients only observe 1–2 specific attack classes (or none at all), local gradients point in conflicting directions, causing destructive interference during linear FedAvg aggregation.

---

## 7. Recommended Next Steps: Iteration 2 (P1 Tasks)

With all P0 tasks completed and verified, the repository is ready for Iteration 2 (P1 tasks) before the November 8 implementation freeze:

1. **`TASK-P1-1` — Distance-Based Robust Aggregation (Multi-Krum):**
   Implement Multi-Krum (Blanchard et al., NeurIPS 2017) in `src/federated/aggregation.py` to complement coordinate-wise median and trimmed mean with a Euclidean distance-based defense.
2. **`TASK-P1-2` — Classical ML Baseline (Random Forest):**
   Train a Random Forest classifier on flattened 11-dimensional feature windows to provide a conventional ML baseline alongside the 1D-CNN.

