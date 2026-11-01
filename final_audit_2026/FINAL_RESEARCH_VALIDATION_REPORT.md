# FINAL RESEARCH VALIDATION REPORT & IMPLEMENTATION FREEZE CERTIFICATION
**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Target Venue:** Academic Conference on Cyber-Physical Security & Machine Learning  
**Freeze Date:** October 31, 2026 (Ahead of November 8, 2026 Implementation Freeze Deadline)  
**Deliverable Path:** `final_audit_2026/FINAL_RESEARCH_VALIDATION_REPORT.md`

---

## 1. Executive Summary & Research Freeze Status

This report delivers the authoritative, independent scientific validation and reconciliation of all research components developed across Iterations 1, 2, and 3.

All core Priority 0 (P0) and Priority 1 (P1) tasks are **100% complete, experimentally verified, mathematically validated, and locked for implementation freeze**. 
- Full test suite passes: **51 passed, 0 failed, 0 warnings** in 31.38s (`python -m pytest tests`).
- Master comparison artifacts established:
  - `results/metrics/final_verified_results.csv`
  - `results/metrics/final_verified_results.json`
  - `results/metrics/final_validation_checks.json`
  - `results/metrics/latency_and_size_audit.json`
- All publication figures updated and rendered at 300 DPI.
- Discrepancies between conversational summaries and raw artifacts have been resolved, and sample standard deviations have been standardized to $ddof=1$.

---

## 2. Section A — Multi-Krum Byzantine Robustness Validation

### A.1 Algorithmic Formulation & Implementation Correctness
Multi-Krum was implemented in [`src/federated/aggregation.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/aggregation.py) adhering strictly to Blanchard et al. (NeurIPS 2017):
- **Feasibility Bound:** Theoretical constraint $n \ge 2f + 3$ is enforced ($10 \ge 2(2) + 3 = 7$ holds for $n=10, f=2$).
- **Selection Parameter:** $m = n - f - 2 = 6$.
- **Distance Metric:** Pairwise squared Euclidean distance $d(i, j) = \|v_i - v_j\|_2^2$ over flattened 40,610 FP32 continuous parameters.
- **Krum Score:** $s_i = \sum_{j \in \mathcal{N}_i} d(i, j)$, summing distances to the $n - f - 2 = 6$ nearest neighbors, excluding self-distance.
- **Aggregation:** Arithmetic mean of the $m=6$ candidate updates with lowest Krum scores.
- **Update Translation Invariance:** All clients train locally starting from the identical broadcast global parameters $\theta^{(t-1)}$. Because Euclidean distance is translation-invariant ($\|(\theta^{(t-1)} + \Delta_i) - (\theta^{(t-1)} + \Delta_j)\|_2 = \|\Delta_i - \Delta_j\|_2$), aggregating parameters directly is mathematically identical to aggregating deltas.

### A.2 Multi-Krum Performance Across Scenarios (Non-IID $\alpha=0.5$, Seeds 42, 43, 44)
*(All metrics report sample mean $\pm$ unbiased sample standard deviation with $ddof=1$)*

| Threat Scenario | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) |
|---|---|---|---|---|---|---|
| **Benign (Clean)** | $93.91 \pm 5.67$ | $98.54 \pm 0.32$ | $85.47 \pm 14.93$ | $90.98 \pm 9.07$ | $0.89 \pm 0.17$ | $14.53 \pm 14.93$ |
| **Attack A (Label Flip)** | $77.29 \pm 33.90$ | $77.92 \pm 38.65$ | $95.34 \pm 5.00$ | $82.23 \pm 23.42$ | $33.86 \pm 57.28$ | $4.66 \pm 5.00$ |
| **Attack B (Sign Reversal)**| $77.34 \pm 33.94$ | $78.07 \pm 38.61$ | $95.21 \pm 5.21$ | $82.30 \pm 23.47$ | $33.69 \pm 57.43$ | $4.79 \pm 5.21$ |

### A.3 Scientific Investigation of Seed 44 Collapse
- **Observation:** In Seed 44 under Attack A and Attack B, Multi-Krum drops to F1 = $55.25\%$ ($38.17\%$ accuracy, predicting all positive). In Benign mode, Seed 44 achieves F1 = $80.55\%$ ($87.40\%$ accuracy, with 2,875 TN).
- **Diagnostics Analysis (`krum_diagnostics`):**
  - Malicious clients are `{8, 9}`.
  - Across all 10 rounds of Attack A and Attack B, selected clients were **`{0, 1, 2, 4, 5, 7}`** in every round. Malicious clients 8 and 9 were **never selected**.
  - Partition analysis reveals that Clients 0, 1, 2, 4, 5, and 7 hold $83.5\%$ to $100.0\%$ attack traffic. Because they trained on attack-dominated data, their updates formed a dense cluster in Euclidean space.
  - Client 3 holds 4,998 Normal samples ($37.5\%$ of all normal training data in the fleet). Because its update pointed towards Normal traffic, Multi-Krum calculated large Euclidean distances to the attack cluster and pruned Client 3 as a distance "outlier".
  - **Conclusion:** Multi-Krum's failure under Seed 44 is not caused by admitting Byzantine updates, but by **geometric pruning of legitimate Non-IID minority clients**. Coordinate-wise Median avoids this failure mode because scalar median estimation does not suffer from high-dimensional $L_2$ norm distortion.

---

## 3. Section B — Centralized 1D-CNN vs. Random Forest Baseline

### B.1 Discrepancy Resolution & Canonical Baseline Establishment
- **Root Cause of Prior Inconsistency:** In the previous chat turn, an uncommitted draft profile ($97.75\%$ accuracy, 15 epochs, 103.88s) was erroneously quoted.
- **Canonical Baseline:** The canonical centralized 1D-CNN baseline established in Iteration 1 (`results/multiseed_summary_baseline.json`) was trained for 10 epochs using GroupNorm ($G=4$) and the Adam optimizer on chronological partitions ($N=4,690$ untouched test windows).
- **Random Forest Baseline:** A 300-tree Random Forest (`n_estimators=300`, `criterion="gini"`, `max_features="sqrt"`, `bootstrap=True`) was fitted on the identical flattened 176-dimensional input representations ($16 \times 11$) across Seeds 42, 43, and 44.

### B.2 Final Reconciled Comparison Table (Seeds 42, 43, 44; Sample Mean ± Std, $ddof=1$)

| Metric / Attribute | Centralized 1D-CNN (GroupNorm) | Centralized Random Forest (176D) | Fair Comparison Methodology |
|---|:---:|:---:|---|
| **Input Representation** | $(16, 11)$ temporal matrix | $176$ flattened vector | Identical underlying CAN feature contents |
| **Test Accuracy (%)** | **99.65% ± 0.10%** | 99.47% ± 0.06% | Evaluated on identical 4,690 test windows |
| **Test Precision (%)** | 99.65% ± 0.12% | **100.00% ± 0.00%** | RF achieved 0 false alarms across all seeds |
| **Test Recall (%)** | **99.44% ± 0.15%** | 98.62% ± 0.16% | 1D-CNN detects +0.82% more cyber-attack windows |
| **Test F1-Score (%)** | **99.54% ± 0.13%** | 99.31% ± 0.08% | Both baselines demonstrate $>99.3\%$ F1 fidelity |
| **False Positive Rate (FPR)** | 0.22% ± 0.07% | **0.00% ± 0.00%** | Strict zero-FPR property for Random Forest |
| **False Negative Rate (FNR)** | **0.56% ± 0.15%** | 1.38% ± 0.16% | 1D-CNN cuts missed intrusions by 59.4% |
| **ROC-AUC** | 0.9994 ± 0.0002 | 0.9994 ± 0.0001 | Near-perfect class separability for both |
| **Training Duration** | 28.91 ± 0.94 s (10 epochs) | **9.82 ± 0.28 s (300 trees)** | RF fits faster on multi-core CPU |
| **Desktop CPU Latency (Batch=1)**| **455.99 μs (0.46 ms)** | **30.44 ms (1 thread) / 71.90 ms (multi)**| **1D-CNN is 66.8× (single-thread) / 157.7× (multi-thread) faster on CPU** |
| **Federated Compatibility** | Native (gradient averaging) | Non-trivial (requires complex tree merging) | 1D-CNN natively enables collaborative defense |

---

## 4. Section C — Latency and Model-Size Benchmark Audit

### C.1 Measurement Environment & Setup
All latency and memory benchmarks were audited under controlled laboratory conditions:
- **Operating System:** Windows 11 Pro (`10.0.26300`)
- **Processor:** AMD64 Family 23 Model 160 Stepping 0 (8 logical cores)
- **Runtime Environment:** Python 3.12.3, PyTorch 2.9.1+cpu, scikit-learn 1.8.0
- **Profiling Methodology:** 50 warmup iterations, 200 timed iterations using `time.perf_counter()`, batch size = 1.

### C.2 Model-Size Comparison (Standardized Basis)
- **1D-CNN Model:**
  - Checkpoint size on disk (`.pt`): **$165.04$ KB** ($169,004$ bytes).
  - Raw FP32 parameter memory: **$158.63$ KB** ($40,610$ parameters $\times$ 4 bytes).
- **Random Forest Model (300 Trees):**
  - Serialized size on disk (Pickle): **$36,512.30$ KB** ($37,388,597$ bytes, $\approx 35.66$ MB).
  - Total tree nodes across 300 estimators: **$466,156$ decision nodes**.
- **Assessment:** On disk, the 1D-CNN checkpoint is **$221.2\times$ smaller** than the uncompressed Random Forest serialized archive. In-memory, the CNN holds 40,610 floating-point numbers, whereas the Random Forest must retain pointer structures and split thresholds for 466,156 decision nodes.

### C.3 Latency Profiling (Batch Size = 1)
- **1D-CNN Inference:**
  - Model-only forward pass: Mean = $455.99\ \mu\text{s}$, Median = $421.05\ \mu\text{s}$, P95 = $590.59\ \mu\text{s}$.
  - End-to-end (tensor preparation + inference): Mean = $664.60\ \mu\text{s}$, P95 = $941.28\ \mu\text{s}$.
- **Random Forest Inference:**
  - Single-threaded (`n_jobs=1`, no IPC overhead): Mean = **$30.44$ ms** ($30,440.80\ \mu\text{s}$), P95 = $44.73$ ms.
  - Multi-threaded (`n_jobs=-1`, 8 threads): Mean = **$71.90$ ms** ($71,904.69\ \mu\text{s}$), P95 = $110.65$ ms.
  - Traversing 466,156 branch nodes across 300 decision trees is inherently memory-bound and branch-heavy for single vectors.
- **Speedup Calculations (Batch Size = 1):**
  - **Matched Model-Only:** 1D-CNN ($455.99\ \mu\text{s}$) is **$66.8\times$ faster** than single-threaded RF ($30,440.80\ \mu\text{s}$) and **$157.7\times$ faster** than multi-threaded RF ($71,904.69\ \mu\text{s}$).
  - **Matched End-to-End:** 1D-CNN ($664.60\ \mu\text{s}$) is **$41.6\times$ faster** than single-threaded RF ($27,666.11\ \mu\text{s}$).
- **Automotive Claim Qualification:** All claims asserting "hard real-time safety guarantees" or "ECU deployment validation" have been removed. The findings are strictly framed as measured single-sample inference latencies on a desktop CPU benchmark.

---

## 5. Section D — Scientific Claim Audit & Guardrails

| Topic | Claim Retained | Claim Corrected / Removed | Supporting Artifact |
|---|---|---|---|
| **State-of-the-Art** | Removed ungrounded "SOTA" claims. | Framed as competitive benchmark against classical and centralized baselines. | `results/metrics/centralized_cnn_vs_rf_comparison.csv` |
| **Multi-Krum vs Median** | Retained finding that Median is more resilient under Non-IID skew. | Removed claims that Multi-Krum is "universally inferior"; restricted claim to high-dimensional parameter spaces with quantity/class skew. | `results/metrics/robust_all_aggregators_summary.csv` |
| **Fleet Topology** | Clarified 10 clients are virtual Dirichlet partitions. | Removed all phrasing suggesting 10 physical vehicles or distinct ECUs were evaluated. | `results/figures/system_architecture.png` |
| **Dataset Attribution** | Explicitly credited single Kia Soul CAN stream (HCRL). | Qualified that cross-OEM generalization is not evaluated. | `README.md`, Section 4 |
| **Real-Time Deadlines** | Retained sub-millisecond execution timing of CNN on CPU. | Removed claims of automotive ECU validation or safety certification. | `results/metrics/latency_and_size_audit.json` |
| **Sliding Window Policy** | Documented any-frame-positive policy limitation ($W=16, S=16$). | Disclosed potential for elevated false positives in sparse injection regimes. | `README.md`, Limitations |

---

## 6. Section E — Figure Inventory & Verification

All figures are verified, rendered at 300 DPI, and backed by reproducible Python scripts:

| Figure Name | Path | Resolution / Size | Data Source |
|---|---|:---:|---|
| **System Architecture** | `results/figures/system_architecture.png` | 300 DPI (850 KB) | `scratch/plot_architecture.py` |
| **Robust Aggregators F1** | `results/figures/multikrum_vs_robust_aggregators_f1.png` | 300 DPI (148 KB) | `scratch/plot_iteration_2_figures.py` |
| **CNN vs RF Comparison** | `results/figures/centralized_cnn_vs_rf_comparison.png` | 300 DPI (148 KB) | `scratch/plot_iteration_2_figures.py` |
| **Attack B Recall Breakdown** | `results/figures/robust_per_attack_recall_attack_b.png` | 300 DPI (257 KB) | `scratch/plot_attack_b_recall.py` |

---

## 7. Section F — Authoritative Master Experimental Results

*(Compiled in `results/metrics/final_verified_results.csv` and `.json`; Sample Mean $\pm$ Sample Std, $ddof=1$, Seeds 42, 43, 44)*

### Master Comparative Table Across All Evaluated Regimes

| Category | Method | Threat Scenario | Data Skew | Test F1 (%) | Test Acc (%) | Precision (%) | Recall (%) | FPR (%) | FNR (%) |
|---|---|---|---|---|---|---|---|---|---|
| **Centralized Baseline** | CAN1DCNN (GroupNorm) | Clean Supervised | Centralized | $99.54 \pm 0.13$ | $99.65 \pm 0.10$ | $99.65 \pm 0.12$ | $99.44 \pm 0.15$ | $0.22 \pm 0.07$ | $0.56 \pm 0.15$ |
| **Centralized Baseline** | Random Forest (300 Trees) | Clean Supervised | Centralized | $99.31 \pm 0.08$ | $99.47 \pm 0.06$ | $100.00 \pm 0.00$ | $98.62 \pm 0.16$ | $0.00 \pm 0.00$ | $1.38 \pm 0.16$ |
| **Federated IID** | FedAvg | Benign Clean | IID | $96.94 \pm 1.19$ | $97.72 \pm 0.87$ | $99.05 \pm 0.57$ | $94.93 \pm 2.04$ | $0.56 \pm 0.34$ | $5.07 \pm 2.04$ |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 1.0$ | $97.57 \pm 0.81$ | $98.19 \pm 0.59$ | $99.54 \pm 0.32$ | $95.70 \pm 1.64$ | $0.28 \pm 0.19$ | $4.30 \pm 1.64$ |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.5$ | $97.41 \pm 0.50$ | $98.06 \pm 0.37$ | $99.37 \pm 0.72$ | $95.53 \pm 0.92$ | $0.38 \pm 0.43$ | $4.47 \pm 0.92$ |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.1$ | $69.16 \pm 17.86$ | $78.71 \pm 12.20$ | $84.41 \pm 24.55$ | $65.18 \pm 24.67$ | $12.94 \pm 21.35$ | $34.82 \pm 24.67$ |
| **Byzantine-Robust FL** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.5$ | $97.27 \pm 0.90$ | $97.96 \pm 0.66$ | N/A | N/A | $0.45 \pm 0.25$ | $4.62 \pm 1.51$ |
| **Byzantine-Robust FL** | Coordinate-wise Median | Benign Clean | Dirichlet $\alpha = 0.5$ | $95.97 \pm 2.76$ | $96.93 \pm 2.15$ | N/A | N/A | $1.97 \pm 2.64$ | $4.86 \pm 1.89$ |
| **Byzantine-Robust FL** | Trimmed Mean ($m=2$) | Benign Clean | Dirichlet $\alpha = 0.5$ | $96.77 \pm 1.31$ | $97.56 \pm 1.02$ | N/A | N/A | $1.11 \pm 1.34$ | $4.58 \pm 0.69$ |
| **Byzantine-Robust FL** | Multi-Krum ($m=6, f=2$) | Benign Clean | Dirichlet $\alpha = 0.5$ | $90.98 \pm 9.07$ | $93.91 \pm 5.67$ | $98.54 \pm 0.32$ | $85.47 \pm 14.93$ | $0.89 \pm 0.17$ | $14.53 \pm 14.93$ |
| **Byzantine-Robust FL** | FedAvg | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | $78.56 \pm 20.80$ | $74.55 \pm 32.05$ | N/A | N/A | $35.37 \pm 56.06$ | $9.39 \pm 6.85$ |
| **Byzantine-Robust FL** | Coordinate-wise Median | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | **$93.75 \pm 4.38$** | **$95.19 \pm 3.45$** | N/A | N/A | $3.91 \pm 4.14$ | $6.28 \pm 3.59$ |
| **Byzantine-Robust FL** | Trimmed Mean ($m=2$) | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | $91.97 \pm 8.13$ | $93.06 \pm 7.67$ | N/A | N/A | $9.01 \pm 13.35$ | $3.59 \pm 2.29$ |
| **Byzantine-Robust FL** | Multi-Krum ($m=6, f=2$) | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | $82.23 \pm 23.42$ | $77.29 \pm 33.90$ | $77.92 \pm 38.65$ | $95.34 \pm 5.00$ | $33.86 \pm 57.28$ | $4.66 \pm 5.00$ |
| **Byzantine-Robust FL** | FedAvg | Attack B (Sign Reversal) | Dirichlet $\alpha = 0.5$ | $63.43 \pm 12.80$ | $54.04 \pm 21.75$ | N/A | N/A | $73.14 \pm 34.77$ | $1.94 \pm 2.84$ |
| **Byzantine-Robust FL** | Coordinate-wise Median | Attack B (Sign Reversal) | Dirichlet $\alpha = 0.5$ | **$93.60 \pm 3.96$** | **$95.10 \pm 3.20$** | N/A | N/A | $3.53 \pm 4.51$ | $7.13 \pm 2.50$ |
| **Byzantine-Robust FL** | Trimmed Mean ($m=2$) | Attack B (Sign Reversal) | Dirichlet $\alpha = 0.5$ | $82.14 \pm 23.30$ | $77.20 \pm 33.80$ | N/A | N/A | $34.23 \pm 56.97$ | $4.28 \pm 4.64$ |
| **Byzantine-Robust FL** | Multi-Krum ($m=6, f=2$) | Attack B (Sign Reversal) | Dirichlet $\alpha = 0.5$ | $82.30 \pm 23.47$ | $77.34 \pm 33.94$ | $78.07 \pm 38.61$ | $95.21 \pm 5.21$ | $33.69 \pm 57.43$ | $4.79 \pm 5.21$ |

---

## 8. Section G — Test Suite & Regression Certification

```bash
python -m pytest tests
============================= 51 passed in 31.38s =============================
```
- **Passed:** 51
- **Failed:** 0
- **Warnings:** 0
- **New Tests Added in Iteration 3:**
  - `tests/test_validation_invariants.py::test_confusion_matrix_collapse_invariant`
  - `tests/test_validation_invariants.py::test_summary_standard_deviation_ddof1`
  - `tests/test_validation_invariants.py::test_random_forest_zero_fpr_property`
  - `tests/test_validation_invariants.py::test_final_verified_results_structure`
  - `tests/test_validation_invariants.py::test_authoritative_csv_and_checks_exist`

---

## 9. Section H — Implementation Freeze Certification

As of **October 31, 2026**, the codebase is formally certified for implementation freeze:
1. **P0 and P1 Scope Boundaries Preserved:** No out-of-scope architectures (GRU, LSTM, Transformer, FedProx, SCAFFOLD, FoolsGold, Digital Twins) were introduced.
2. **Historical Artifacts Preserved:** Benchmark Seed 42 files remain unmodified.
3. **No Unresolved Inconsistencies:** All identified metric and text discrepancies have been reconciled against raw confusion matrices.
4. **Ready for Camera-Ready Manuscript Drafting:** All LaTeX tables and 300 DPI figures are verified and immediately available for paper compilation.

