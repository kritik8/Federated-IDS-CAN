# Phase 3A — Independent Experimental Audit and Results Freeze Report

**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Repository:** `https://github.com/kritik8/Federated-IDS-CAN.git`  
**Base Commit Audited:** `03d1079` (`feat: evaluate robust aggregation under client attacks`)  
**Audit Date:** October 10, 2026  
**Audit Status:** **VERIFIED WITH RECONCILED DOCUMENTATION**

---

## 1. Audit Scope & Environment

An independent experimental reproducibility and scientific-consistency audit was conducted across all Phase 3 robust federated learning experiments, model checkpoints, metric JSON files, evaluation notebooks, and reports.

### Software & Hardware Environment
- **Operating System:** Windows 11 (AMD64)
- **Python Version:** 3.12.3 (v3.12.3:f6650f9, Apr 9 2024, MSC v.1938 64 bit)
- **PyTorch:** 2.9.1+cpu (Deterministic CPU execution)
- **NumPy:** 2.3.1
- **Scikit-Learn:** 1.8.0
- **Random Seeds:** 42 (Primary), 43, 44

---

## 2. Three-Seed Experiment Matrix & Protocol Verification

We audited whether the experiments across seeds 42, 43, and 44 represent genuine independent executions.

### 2.1 Experiment Matrix

| Configuration ID | Seed | Partition Basis | Malicious Client IDs | Aggregator | Best Round | Checkpoint Path | JSON Metric Path |
|---|:---:|---|:---:|---|:---:|---|---|
| `benign_fedavg` | 42 | Dirichlet ($\alpha=0.5$, s=42) | None | FedAvg | Rd 10 | `results/models/robust_benign_fedavg_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `benign_median` | 42 | Dirichlet ($\alpha=0.5$, s=42) | None | Median | Rd 10 | `results/models/robust_benign_median_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `benign_trimmed_mean`| 42 | Dirichlet ($\alpha=0.5$, s=42) | None | Trimmed Mean ($m=2$) | Rd 10 | `results/models/robust_benign_trimmed_mean_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `label_flip_fedavg` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | FedAvg | Rd 10 | `results/models/robust_label_flip_fedavg_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `label_flip_median` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | Median | Rd 9 | `results/models/robust_label_flip_median_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `label_flip_trimmed_mean` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | Trimmed Mean ($m=2$) | Rd 9 | `results/models/robust_label_flip_trimmed_mean_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `update_corrupt_fedavg` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | FedAvg | Rd 2 | `results/models/robust_update_corrupt_fedavg_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `update_corrupt_median` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | Median | Rd 9 | `results/models/robust_update_corrupt_median_best.pt` | `results/metrics/robust_federated_experiments.json` |
| `update_corrupt_trimmed_mean` | 42 | Dirichlet ($\alpha=0.5$, s=42) | [8, 9] | Trimmed Mean ($m=2$) | Rd 10 | `results/models/robust_update_corrupt_trimmed_mean_best.pt` | `results/metrics/robust_federated_experiments.json` |
| Multi-Seed Runs (18 runs) | 43, 44 | Dirichlet ($\alpha=0.5$, s=43, 44) | [8, 9] | All 3 aggregators | Varied | Dynamically evaluated during multi-seed run | `results/metrics/robust_multiseed_summary.json` |

### 2.2 Protocol Audit Findings:
1. **Fresh Models:** Every run instantiated a fresh `CAN1DCNN` model initialized deterministically with `torch.manual_seed(seed)` in `FederatedServer`.
2. **Partition Independence & Coverage:**
   - Seed 42: Allocated 21,875 samples across 10 clients. Disjoint and complete: **True** (Union = 21,875, unique = 21,875).
   - Seed 43: Allocated 21,875 samples across 10 clients. Disjoint and complete: **True** (Union = 21,875, unique = 21,875).
   - Seed 44: Allocated 21,875 samples across 10 clients. Disjoint and complete: **True** (Union = 21,875, unique = 21,875).
3. **Partition Uniformity within Matched Comparisons:** For a given seed, all 9 regimes shared the identical partition and identical malicious clients (`client_8` and `client_9`).
4. **Byzantine Budget:** Always exactly 2 out of 10 clients ($f=20\%$).
5. **No Test Leakage:** Checkpoints were tracked and selected strictly using Validation F1 on the untouched validation set ($N=4,685$). The test set ($N=4,690$) was evaluated only post-selection.
6. **Bit-for-Bit Deterministic Reproducibility:** Re-executed `update_corrupt_fedavg` on Seed 43 from raw data. Resulting Test F1 was `0.781806` (difference: `0.000000`).

---

## 3. Metric Consistency & Checkpoint Re-evaluation

We loaded all 9 primary checkpoints from `results/models/robust_*_best.pt` and executed an independent inference audit against `data/splits/test_split.npz` (4,690 samples: 2,900 Normal, 1,790 Attack).

### 3.1 Checkpoint Recomputation vs. Stored Metrics

| Scenario | Recomputed CM [TN, FP, FN, TP] | Stored CM [TN, FP, FN, TP] | Recomputed Acc (%) | Stored Acc (%) | Recomputed F1 (%) | Stored F1 (%) | Recomputed AUC | Stored AUC | Match Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `benign_fedavg` | [2893, 7, 53, 1737] | [2893, 7, 53, 1737] | 98.72% | 98.72% | 98.30% | 98.30% | 0.9964 | 0.9964 | **EXACT MATCH** |
| `benign_median` | [2879, 21, 49, 1741] | [2879, 21, 49, 1741] | 98.51% | 98.51% | 98.03% | 98.03% | 0.9956 | 0.9956 | **EXACT MATCH** |
| `benign_trimmed_mean` | [2890, 10, 68, 1722] | [2890, 10, 68, 1722] | 98.34% | 98.34% | 97.79% | 97.79% | 0.9955 | 0.9955 | **EXACT MATCH** |
| `label_flip_fedavg` | [2898, 2, 250, 1540] | [2898, 2, 250, 1540] | 94.63% | 94.63% | 92.44% | 92.44% | 0.9932 | 0.9932 | **EXACT MATCH** |
| `label_flip_median` | [2809, 91, 43, 1747] | [2809, 91, 43, 1747] | 97.14% | 97.14% | 96.31% | 96.31% | 0.9945 | 0.9945 | **EXACT MATCH** |
| `label_flip_trimmed_mean`| [2831, 69, 48, 1742] | [2831, 69, 48, 1742] | 97.51% | 97.51% | 96.75% | 96.75% | 0.9946 | 0.9946 | **EXACT MATCH** |
| `update_corrupt_fedavg` | [0, 2900, 0, 1790] | [0, 2900, 0, 1790] | 38.17% | 38.17% | 55.25% | 55.25% | 0.5861 | 0.5861 | **EXACT MATCH** |
| `update_corrupt_median` | [2850, 50, 76, 1714] | [2850, 50, 76, 1714] | 97.31% | 97.31% | 96.45% | 96.45% | 0.9934 | 0.9934 | **EXACT MATCH** |
| `update_corrupt_trimmed_mean` | [2827, 73, 65, 1725] | [2827, 73, 65, 1725] | 97.06% | 97.06% | 96.15% | 96.15% | 0.9936 | 0.9936 | **EXACT MATCH** |

### 3.2 Mathematical Verification of the Update-Corruption FedAvg Collapse
When `update_corrupt_fedavg` collapsed, every single one of the 4,690 test samples was classified as Attack ($\hat{y}=1$):
- $\text{TN} = 0, \quad \text{FP} = 2,900, \quad \text{FN} = 0, \quad \text{TP} = 1,790$
- $\text{Accuracy} = \frac{1,790 + 0}{4,690} = 38.1663\% \approx 38.17\%$
- $\text{Precision} = \frac{1,790}{1,790 + 2,900} = 38.1663\% \approx 38.17\%$
- $\text{Recall} = \frac{1,790}{1,790 + 0} = 100.00\%$
- $\text{F1} = 2 \times \frac{0.381663 \times 1.0}{0.381663 + 1.0} = 55.2469\% \approx 55.25\%$
- $\text{FPR} = \frac{2,900}{2,900 + 0} = 100.00\%$
- $\text{FNR} = \frac{0}{0 + 1,790} = 0.00\%$
- $\text{ROC-AUC} = 0.5861$ based on the continuous softmax probability scores (which all fell in range $p(y=1) \in [0.55, 0.99]$). Under discrete thresholding, ROC-AUC is $0.5000$.

The metrics are **100% mathematically consistent**.

### 3.3 Recomputed Per-Attack Detection Breakdown (Seed 42)
Recomputed directly from `y_types_test` metadata:

| Scenario | Normal Acc (2,900) | DoS Recall (398) | Fuzzy Recall (209) | Gear Recall (591) | RPM Recall (592) | Overall Recall (1,790) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `benign_fedavg` | 99.76% (2893/2900) | 98.24% (391/398) | 79.90% (167/209) | 99.32% (587/591) | 100.00% (592/592) | 97.04% (1737/1790) |
| `benign_median` | 99.28% (2879/2900) | 98.24% (391/398) | 81.82% (171/209) | 99.32% (587/591) | 100.00% (592/592) | 97.26% (1741/1790) |
| `benign_trimmed_mean` | 99.66% (2890/2900) | 97.99% (390/398) | 77.51% (162/209) | 98.14% (580/591) | 99.66% (590/592) | 96.20% (1722/1790) |
| `label_flip_fedavg` | 99.93% (2898/2900) | **96.23% (383/398)** | **81.82% (171/209)** | **94.08% (556/591)** | **72.64% (430/592) ⚠️** | **86.03% (1540/1790) ⚠️** |
| `label_flip_median` | 96.86% (2809/2900) | 98.24% (391/398) | 85.65% (179/209) | 98.98% (585/591) | 100.00% (592/592) | 97.60% (1747/1790) |
| `label_flip_trimmed_mean`| 97.62% (2831/2900) | 97.99% (390/398) | 85.65% (179/209) | 98.48% (582/591) | 99.83% (591/592) | 97.32% (1742/1790) |
| `update_corrupt_fedavg` | 0.00% (0/2900) | 100.00% (398/398) | 100.00% (209/209) | 100.00% (591/591) | 100.00% (592/592) | 100.00% (1790/1790)* |
| `update_corrupt_median` | 98.28% (2850/2900) | 97.74% (389/398) | 78.47% (164/209) | 97.46% (576/591) | 98.82% (585/592) | 95.75% (1714/1790) |
| `update_corrupt_trimmed_mean` | 97.48% (2827/2900) | 98.24% (391/398) | 80.86% (169/209) | 98.14% (580/591) | 98.82% (585/592) | 96.37% (1725/1790) |

*\*Note: Under update corruption, FedAvg achieved 100% recall only due to total collapse (FPR = 100.00%), predicting all 4,690 frames as attacks.*

---

## 4. Multi-Seed Statistical Summary Audit

We audited the multi-seed summary metrics stored in `results/metrics/robust_multiseed_summary.json`:
- **Convention:** Standard deviation in `robust_multiseed_summary.json` uses **`ddof=0` (population standard deviation)**: `np.std(x, ddof=0)`.
- Below, we report both `ddof=0` and `ddof=1` (sample standard deviation) alongside the exact per-seed metrics:

### 4.1 Verified Per-Seed Metrics Table

| Scenario | Seed 42 F1 (%) | Seed 43 F1 (%) | Seed 44 F1 (%) | Mean F1 (%) | Std F1 (`ddof=0`) | Std F1 (`ddof=1`) | Mean Acc (%) | Mean FPR (%) | Mean FNR (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `benign_fedavg` | 98.30% | 96.64% | 96.87% | **97.27%** | $\pm$ 0.73% | $\pm$ 0.90% | 97.96% | 0.45% | 4.62% |
| `benign_median` | 98.03% | 97.05% | 92.83% | **95.97%** | $\pm$ 2.26% | $\pm$ 2.76% | 96.93% | 1.97% | 4.86% |
| `benign_trimmed_mean` | 97.79% | 97.23% | 95.29% | **96.77%** | $\pm$ 1.07% | $\pm$ 1.31% | 97.56% | 1.11% | 4.58% |
| `label_flip_fedavg` | 92.44% | 88.61% | 54.64% | **78.56%** | $\pm$ 16.99% | $\pm$ 20.80% | 74.55% | 35.37% | 9.39% |
| `label_flip_median` | 96.31% | 96.24% | 88.69% | **93.75%** | $\pm$ 3.57% | $\pm$ 4.38% | 95.19% | 3.91% | 6.28% |
| `label_flip_trimmed_mean` | 96.75% | 96.58% | 82.58% | **91.97%** | $\pm$ 6.64% | $\pm$ 8.13% | 93.06% | 9.01% | 3.59% |
| `update_corrupt_fedavg` | 55.25% | 78.18% | 56.87% | **63.43%** | $\pm$ 10.45% | $\pm$ 12.80% | 54.04% | 73.14% | 1.94% |
| `update_corrupt_median` | 96.45% | 95.28% | 89.08% | **93.60%** | $\pm$ 3.24% | $\pm$ 3.96% | 95.10% | 3.53% | 7.13% |
| `update_corrupt_trimmed_mean` | 96.15% | 95.03% | 55.25% | **82.14%** | $\pm$ 19.02% | $\pm$ 23.30% | 77.20% | 34.23% | 4.28% |

---

## 5. Implementation Code Review Findings

### 5.1 Sample-Weighted FedAvg
- Verified in [`src/federated/aggregation.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/aggregation.py#L71):
  $$w_k = \frac{n_k}{\sum_{j} n_j}, \quad \theta_{\text{global}} = \sum_{k=1}^K w_k \theta_k$$
  Weights are proportional to local client sample size.

### 5.2 Coordinate-wise Median
- Verified in [`src/federated/aggregation.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/aggregation.py#L107):
  Stacked tensors along dimension 0, computed via `torch.quantile(stacked, q=0.5, dim=0)`.
  Interpolates linearly between central coordinates for even $K$. Unweighted by client sample count to prevent malicious weight spoofing.

### 5.3 Coordinate-wise Trimmed Mean
- Verified in [`src/federated/aggregation.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/aggregation.py#L147):
  Tensors sorted along dimension 0. Trims $m=2$ extreme values on each side:
  $$\text{retained} = \text{sorted}[m : K - m] = \text{sorted}[2 : 8]$$
  Averages the middle 6 updates. Validates $m \ge 0$ and $2m < K$.

### 5.4 Threat Model: Delta Sign Reversal vs Full State
- Verified in [`src/federated/threat.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/threat.py#L55):
  $$\Delta_k = \theta_k - \theta_{\text{global}}, \quad \theta_k' = \theta_{\text{global}} - \gamma \Delta_k$$
  Operates strictly on the update delta relative to the global broadcast model, preserving architectural parameters and correctly reversing gradient direction.

### 5.5 Threat Model: Label Flipping Isolation
- Verified in [`src/federated/threat.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/threat.py#L17):
  Applied strictly to malicious client local training partitions ($y \mapsto 1-y$).
  Benign client labels, validation labels, and test labels are never poisoned.

### 5.6 GroupNorm Compatibility
- The `CAN1DCNN` utilizes `GroupNorm(4, C)` with `weight` and `bias` parameters.
- `validate_client_updates` verifies floating-point dtype and identical shape across all client keys. No running statistics exist to corrupt.

---

## 6. Documentation Discrepancies & Resolutions

During cross-artifact reconciliation, three documentation discrepancies were identified and resolved:

### Discrepancy 1: Malicious Client Identities and $\gamma$ Scaling Factor in `README.md`
- **Issue:** `README.md` stated malicious clients were `client_0` and `client_1` with $\gamma=1.5$.
- **Ground Truth:** `configs/robust_federated_config.json`, `results/metrics/robust_federated_experiments.json`, and `results/reports/robust_federated_report.md` all used `client_8` and `client_9` with $\gamma=1.0$ (holding 3,572 and 5,101 samples respectively).
- **Resolution:** Reconciled `README.md` to match the exact configured parameters `[8, 9]` and $\gamma=1.0$.

### Discrepancy 2: Multi-Seed Summary Table in `README.md`
- **Issue:** The multi-seed table in `README.md` contained values from an early draft (e.g. `Label Flip FedAvg` Accuracy: 85.90%, FNR: 31.25%).
- **Ground Truth:** `results/metrics/robust_multiseed_summary.json` and `results/reports/robust_federated_report.md` record `Label Flip FedAvg` Accuracy as $74.55\% \pm 26.17\%$ and FNR as $9.39\% \pm 5.59\%$ (`ddof=0`).
- **Resolution:** Updated `README.md` multi-seed table to match the verified JSON ground truth.

### Discrepancy 3: Per-Attack Breakdown Table for `label_flip_fedavg`
- **Issue:** `robust_federated_report.md` Section 7 table reported DoS at 73.62% and RPM at 95.95% for `label_flip_fedavg`.
- **Ground Truth:** The saved checkpoint `results/models/robust_label_flip_fedavg_best.pt`, `notebooks/06_robust_federated_learning.ipynb`, and `results/metrics/robust_federated_experiments.json` all show:
  DoS: 96.23% (383/398), Fuzzy: 81.82% (171/209), Gear: 94.08% (556/591), and RPM: 72.64% (430/592) with total TP = 1,540 (Recall = 86.03%).
- **Resolution:** Reconciled `robust_federated_report.md` Section 7 and `README.md` to reflect the verified checkpoint evaluation.

---

## 7. Remaining Scientific Limitations

1. **Simulated vs. Physical Vehicular Fleet:** The 10 clients are simulated partitions of a single Kia Soul dataset rather than 10 physical vehicles with distinct ECU hardware, clock drifts, and wiring bus lengths.
2. **Threat Model Scope:** Evaluated untargeted binary label flipping and model-update delta sign reversal. Advanced targeted backdoors (e.g., bit triggers in payload bytes that activate only upon specific arbitration sequences) require spectral or cosine-similarity defenses (e.g., FoolsGold).
3. **Trimmed Mean Non-IID Fragility:** In extreme Non-IID fleet settings ($\alpha=0.5$), legitimate specialized clients frequently appear in the upper or lower $m$ tails of coordinate distributions, leading Trimmed Mean to discard honest updates.
4. **Absence of Cryptographic Privacy:** Parameter exchange alone does not prevent reconstruction or membership inference attacks; differential privacy ($\epsilon, \delta$) or secure multiparty computation is required for formal privacy.

---

## 8. Final Audit Status

# **VERIFIED WITH RECONCILED DOCUMENTATION**

All 35 unit tests pass. All saved checkpoints produce exact matches with stored JSON metrics. All three seeds were genuine independent runs. All documentation discrepancies have been documented and corrected. The experimental results are frozen and ready for citation in the research paper.
