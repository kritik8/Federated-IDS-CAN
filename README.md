# Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-53%20Passed%20(0%20Warnings)-brightgreen?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Implementation-Frozen%20(Nov%201%2C%202026)-success?style=flat-square)](final_audit_2026/FINAL_IMPLEMENTATION_FREEZE_REPORT.md)

**A rigorous, reproducible research framework evaluating decentralized Federated Learning under fleet heterogeneity (Non-IID traffic) and adversarial Byzantine client poisoning for in-vehicle Controller Area Network (CAN) security.**

[Overview](#-overview) • [System Architecture](#-system-architecture) • [Authoritative Results](#-authoritative-master-experimental-results) • [Key Scientific Insights](#-key-scientific-insights) • [Threat Models & Defenses](#-threat-models--defense-mechanisms) • [Scope & Guardrails](#-scientific-guardrails--limitations) • [Quickstart & Reproducibility](#-quickstart--reproducibility) • [Repository Structure](#-repository-structure)

</div>

---

## 📌 Overview

Modern passenger vehicles integrate over 70 Electronic Control Units (ECUs) exchanging sensor telemetry and actuation commands across the internal **Controller Area Network (CAN)** bus. Because the classical CAN protocol lacks native encryption, message authentication, or sender identity verification, a compromised gateway (e.g., via infotainment, cellular telematics, or OBD-II diagnostics) can inject arbitrary arbitration IDs and malicious payloads with high bus priority.

### Why Federated Learning for CAN IDS?
1. **Bandwidth Preservation:** Streaming millions of high-frequency raw CAN frames from connected vehicle fleets to centralized cloud repositories introduces extreme communication and storage bottlenecks.
2. **Confidentiality by Data Locality:** Federated Learning (FL) coordinates distributed model training across edge clients (vehicles or subsystem gateways) by exchanging local parameter updates ($\theta_k$) rather than transmitting raw packet streams.
3. **The Core Vulnerability:** In decentralized fleets, standard Federated Averaging (`FedAvg`) is fundamentally fragile against two prevalent vehicular challenges:
   - **Fleet Traffic Heterogeneity (Non-IID Drift):** Varying vehicle models, driving behaviors, and localized cyber-attacks cause severe client drift, destabilizing global model convergence.
   - **Byzantine Adversarial Poisoning:** Compromised or rogue vehicles participating in collaborative learning can submit crafted or inverted model updates to blind the global IDS or trigger denial-of-service false alarms.

This repository provides an end-to-end, scientifically validated benchmark that evaluates lightweight edge deep learning architectures, quantifies FedAvg degradation under Dirichlet data skew ($\alpha \in \{1.0, 0.5, 0.1\}$), and benchmarks Byzantine-robust aggregation mechanisms (**Coordinate-wise Median**, **Coordinate-wise Trimmed Mean**, and distance-based **Multi-Krum**) against data poisoning (Label Flipping) and model poisoning (Sign Reversal).

---

## 🏗️ System Architecture

The pipeline processes raw CAN packet captures into sliding temporal windows, trains edge-optimized neural classifiers with Group Normalization, simulates heterogeneous client partitions, and aggregates updates under Byzantine threat models:

```text
[Raw CAN Telemetry Stream] (HCRL Kia Soul: Normal, DoS, Fuzzy, Gear, RPM)
       │
       ▼
[Feature Extraction & Windowing]
       ├── Dynamic Variable DLC Parsing (Handles DLC < 8 without byte corruption)
       ├── 11-Dimensional Feature Vector: Normalized CAN ID, DLC, 8 Data Bytes, Domain-Scaled Δt
       └── Non-Overlapping Sliding Windows (Window Size W=16, Stride S=16)
       │
       ▼
[Chronological Scenario Split] (Strictly per scenario: 70% Train, 15% Val, 15% Test)
       │
       ▼
[Edge-Optimized Neural Model: CAN1DCNN]
       ├── 3 Conv1D Blocks + GroupNorm (G=4) + ReLU + MaxPool (40,610 FP32 Parameters, ~165 KB)
       └── Replaces BatchNorm1d to eliminate running mean/variance divergence under Non-IID data
       │
       ▼
[Federated Fleet Simulation] (10 Virtual Clients via Dirichlet Allocation: α ∈ {1.0, 0.5, 0.1})
       │
       ├── Honest Clients (k ∈ {0, 1, ..., 7}): Local SGD on Private Partitions
       └── Byzantine Clients (k ∈ {8, 9}, 20% by count, 25%-40% by training volume):
             ├── Attack A (Data Poisoning): Label Flipping (y ↦ 1 - y)
             └── Attack B (Model Poisoning): Update Sign Reversal (Δk' = -γ Δk, γ = 1.0)
       │
       ▼
[Central Aggregation Server]
       ├── Vanilla Baseline: Sample-Weighted Federated Averaging (FedAvg)
       ├── Coordinate-wise Median: θj = median({θk,j})
       ├── Coordinate-wise Trimmed Mean: θj = trimmed_mean({θk,j}, m=2)
       └── Distance-Based Multi-Krum: Candidate selection via Euclidean distance neighborhood scores
```

A publication-ready, 300-DPI architectural diagram is available at [`results/figures/system_architecture.png`](results/figures/system_architecture.png).

---

## 📊 Authoritative Master Experimental Results

All reported metrics are computed on the held-out test partition ($N=4,690$ windows: 2,900 Normal, 1,790 Attack) across **Seeds 42, 43, and 44** using sample mean and unbiased sample standard deviation ($ddof=1$). Raw results are compiled in [`results/metrics/final_verified_results.csv`](results/metrics/final_verified_results.csv).

| Category | Method / Model | Threat Scenario | Data Skew | Test F1 (%) | Test Acc (%) | Precision (%) | Recall (%) | FPR (%) | FNR (%) |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | CAN1DCNN (GroupNorm) | Clean Supervised | Centralized | **99.54 ± 0.13** | **99.65 ± 0.10** | 99.65 ± 0.12 | **99.44 ± 0.15** | 0.22 ± 0.07 | **0.56 ± 0.15** |
| **Centralized Baseline** | Random Forest (300 Trees) | Clean Supervised | Centralized | 99.31 ± 0.08 | 99.47 ± 0.06 | **100.00 ± 0.00** | 98.62 ± 0.16 | **0.00 ± 0.00** | 1.38 ± 0.16 |
| **Federated IID** | FedAvg | Benign Clean | IID Uniform | 96.94 ± 1.19 | 97.72 ± 0.87 | 99.05 ± 0.57 | 94.93 ± 2.04 | 0.56 ± 0.34 | 5.07 ± 2.04 |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 1.0$ | 97.57 ± 0.81 | 98.19 ± 0.59 | 99.54 ± 0.32 | 95.70 ± 1.64 | 0.28 ± 0.19 | 4.30 ± 1.64 |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.5$ | 97.41 ± 0.50 | 98.06 ± 0.37 | 99.37 ± 0.72 | 95.53 ± 0.92 | 0.38 ± 0.43 | 4.47 ± 0.92 |
| **Federated Non-IID** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.1$ | **69.16 ± 17.86** | **78.71 ± 12.20** | 84.41 ± 24.55 | 65.18 ± 24.67 | 12.94 ± 21.35 | 34.82 ± 24.67 |
| **Robust Federated** | FedAvg | Benign Clean | Dirichlet $\alpha = 0.5$ | 97.27 ± 0.90 | 97.96 ± 0.66 | N/A | N/A | 0.45 ± 0.25 | 4.62 ± 1.51 |
| **Robust Federated** | Coordinate-wise Median | Benign Clean | Dirichlet $\alpha = 0.5$ | 95.97 ± 2.76 | 96.93 ± 2.15 | N/A | N/A | 1.97 ± 2.64 | 4.86 ± 1.89 |
| **Robust Federated** | Trimmed Mean ($m=2$) | Benign Clean | Dirichlet $\alpha = 0.5$ | 96.77 ± 1.31 | 97.56 ± 1.02 | N/A | N/A | 1.11 ± 1.34 | 4.58 ± 0.69 |
| **Robust Federated** | Multi-Krum ($m=6, f=2$) | Benign Clean | Dirichlet $\alpha = 0.5$ | 90.98 ± 9.07 | 93.91 ± 5.67 | 98.54 ± 0.32 | 85.47 ± 14.93 | 0.89 ± 0.17 | 14.53 ± 14.93 |
| **Robust Federated** | FedAvg | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | 78.56 ± 20.80 | 74.55 ± 32.05 | N/A | N/A | 35.37 ± 56.06 | 9.39 ± 6.85 |
| **Robust Federated** | Coordinate-wise Median | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | **93.75 ± 4.38** | **95.19 ± 3.45** | N/A | N/A | **3.91 ± 4.14** | **6.28 ± 3.59** |
| **Robust Federated** | Trimmed Mean ($m=2$) | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | 91.97 ± 8.13 | 93.06 ± 7.67 | N/A | N/A | 9.01 ± 13.35 | 3.59 ± 2.29 |
| **Robust Federated** | Multi-Krum ($m=6, f=2$) | Attack A (Label Flip) | Dirichlet $\alpha = 0.5$ | 82.23 ± 23.42 | 77.29 ± 33.90 | 77.92 ± 38.65 | 95.34 ± 5.00 | 33.86 ± 57.28 | 4.66 ± 5.00 |
| **Robust Federated** | FedAvg | Attack B (Sign Reversal)| Dirichlet $\alpha = 0.5$ | 63.43 ± 12.80 | 54.04 ± 21.75 | N/A | N/A | 73.14 ± 34.77 | 1.94 ± 2.84 |
| **Robust Federated** | Coordinate-wise Median | Attack B (Sign Reversal)| Dirichlet $\alpha = 0.5$ | **93.60 ± 3.96** | **95.10 ± 3.20** | N/A | N/A | **3.53 ± 4.51** | **7.13 ± 2.50** |
| **Robust Federated** | Trimmed Mean ($m=2$) | Attack B (Sign Reversal)| Dirichlet $\alpha = 0.5$ | 82.14 ± 23.30 | 77.20 ± 33.80 | N/A | N/A | 34.23 ± 56.97 | 4.28 ± 4.64 |
| **Robust Federated** | Multi-Krum ($m=6, f=2$) | Attack B (Sign Reversal)| Dirichlet $\alpha = 0.5$ | 82.30 ± 23.47 | 77.34 ± 33.94 | 78.07 ± 38.61 | 95.21 ± 5.21 | 33.69 ± 57.43 | 4.79 ± 5.21 |

---

## 🔍 Key Scientific Insights

### 1. Robust Defense Superiority: Coordinate-wise Median vs. Distance-Based Multi-Krum
- **Coordinate-wise Median** demonstrates the highest resilience across both adversarial threat vectors, achieving **93.75% F1** under Attack A (Label Flipping) and **93.60% F1** under Attack B (Sign Reversal). Because it computes the median independently across every scalar weight coordinate, it is naturally immune to extreme coordinate perturbations and tolerates up to $f < K/2$ arbitrary Byzantine clients.
- **Multi-Krum Vulnerability Under Fleet Skew:** Although classical Byzantine theory guarantees Multi-Krum convergence under IID assumptions, in heterogeneous vehicular networks (Dirichlet $\alpha=0.5$), Multi-Krum degrades to **$82.23\%-82.30\%$ F1**. Forensic inspection revealed that Multi-Krum's failure is driven by **geometric pruning of honest minority clients**: honest clients observing specialized attack or normal traffic naturally produce parameter updates distant from the majority cluster. Multi-Krum misidentifies this statistical heterogeneity as malicious behavior, discarding legitimate updates while admitting clustered majority updates.

### 2. Client Drift Under Extreme Non-IID Traffic ($\alpha = 0.1$)
- Under moderate Dirichlet skew ($\alpha = 1.0$ and $\alpha = 0.5$), FedAvg maintains high detection capability ($97.57\%$ and $97.41\%$ F1).
- Under extreme class skew ($\alpha = 0.1$), FedAvg collapses to **$69.16\% \pm 17.86\%$ F1** ($78.71\% \pm 12.20\%$ Accuracy). Forensic analysis of per-seed runs demonstrates the physical reality of vehicular client drift:
  - **Seed 43:** Severe label starvation induces classifier bias; the model achieves $58.60\%$ F1 with an extreme **63.24% False Negative Rate**, missing nearly two-thirds of safety-critical intrusions.
  - **Seed 44:** Inverted class skew biases the global model toward false alarms, resulting in a **37.59% False Positive Rate**.

### 3. Edge Latency and Computational Footprint
Audited single-sample inference latency ($B=1$) on an AMD64 desktop CPU (`AMD64 Family 23 Model 160 Stepping 0`, 8 cores, Windows 11) using batch size 1 under a protocol of 50 warmup + 200 timed iterations:

| Feature / Metric | Proposed CAN1DCNN | Random Forest (300 Trees) | Comparative Efficiency |
|---|:---:|:---:|:---:|
| **Trainable Parameters** | 40,610 FP32 weights | 466,156 tree nodes | **11.5× fewer decision parameters** |
| **On-Disk Checkpoint Size** | **165.26 KB** (`.pt`) | **36,558.15 KB** (`.joblib`) | **221.2× smaller storage footprint** |
| **Model-Only Latency (Single-Thread)** | **455.99 μs** (0.456 ms) | 30,440.80 μs (30.44 ms) | **CAN1DCNN is 66.8× faster** |
| **Model-Only Latency (Multi-Thread)** | **455.99 μs** (0.456 ms) | 71,904.69 μs (71.90 ms) | **CAN1DCNN is 157.7× faster** |
| **Decentralized FL Readiness** | **Native Gradient Exchange** | Non-trivial / Infeasible | Supports collaborative learning |

> **Architectural Performance Driver:** Random Forest requires sequential or parallel traversal across 466,156 nodes, leading to frequent CPU cache misses and branch mispredictions. On a single $1 \times 176$ input vector, threadpool synchronization overhead doubles multi-threaded RF latency ($30.44\text{ ms} \to 71.90\text{ ms}$). In contrast, CAN1DCNN executes contiguous matrix multiply-accumulate (GEMM) operations accelerated by vectorized CPU instructions.
>
> *(Note: Due to high inter-loop timing variance in separate benchmark loops, end-to-end Random Forest latency is marked as unverified and omitted from comparative claims. Comparisons strictly reflect verified model-only inference latencies).*

---

## ⚔️ Threat Models & Defense Mechanisms

### 1. Adversary Fleet Budget
- 10 simulated vehicular clients, 10 communication rounds, Dirichlet $\alpha = 0.5$.
- **2 out of 10 clients ($f=2$, 20% by client count)** are designated Byzantine attackers (`client_8` and `client_9`).
- **Dirichlet Quantity Skew:** Due to non-uniform sample allocation, `client_8` and `client_9` control **$24.99\%$ to $39.65\%$ of all training samples** across seeds:
  - **Seed 42:** 8,673 poisoned samples (**39.65%** of training pool).
  - **Seed 43:** 5,467 poisoned samples (**24.99%** of training pool).
  - **Seed 44:** 8,366 poisoned samples (**38.24%** of training pool).

### 2. Evaluated Threat Vectors
- **Attack A — Label Flipping (Data Poisoning):** Byzantine clients invert binary labels during local SGD ($y \mapsto 1 - y$). This pulls the global decision boundary to classify attack frames as normal and benign frames as intrusions.
- **Attack B — Model-Update Sign Reversal (Byzantine Model Poisoning):** Byzantine clients compute legitimate local gradients but transmit inverted parameter updates:
  $$\Delta_k' = -\gamma \cdot \Delta_k \quad (\gamma = 1.0) \implies \theta_k' = 2\theta^{(t-1)} - \theta_k^{(t)}$$
  Under FedAvg, this causes complete global model collapse ($54.04\%$ Accuracy, $73.14\%$ FPR).

### 3. Aggregation Rules
- **FedAvg:** $\theta^{(t)} = \sum_{k} \frac{n_k}{N} \theta_k^{(t)}$ (Baseline linear weighted average).
- **Coordinate-wise Median:** $\theta_j^{(t)} = \text{median}(\{\theta_{k, j}^{(t)}\}_{k=1}^K)$ for each weight coordinate $j$.
- **Coordinate-wise Trimmed Mean:** $\theta_j^{(t)} = \frac{1}{K - 2m} \sum_{k=m+1}^{K-m} \theta_{(k), j}^{(t)}$ ($m=2$).
- **Multi-Krum:** Computes pairwise Euclidean distance sums over $n - f - 2 = 6$ nearest neighbors and averages the top $m=6$ candidate models.

---

## ⚠️ Scientific Guardrails & Limitations

To preserve strict academic integrity, all findings are bound by the following scope declarations:
1. **Single-Vehicle Telemetry:** All experiments evaluate the HCRL Car-Hacking Dataset recorded from a single commercial Kia Soul vehicle.
2. **Simulated Virtual Partitions:** The 10 federated clients are simulated partitions of one Kia Soul dataset, not actual physical vehicles or validated ECUs. They model virtual client partitions to assess algorithmic behavior under data heterogeneity.
3. **Window Labelling Policy:** Non-overlapping contiguous 16-frame windows ($W=16, S=16$) are labeled as Attack ($y=1$) if at least one frame within the window is malicious (any-frame-positive). In sparse injection scenarios, this can slightly elevate false positive rates.
4. **Privacy Scope:** Raw CAN records are not directly exchanged between clients and server; however, local model updates may leak information, and no formal differential privacy or cryptographic guarantee is provided.
5. **Cross-OEM Generalization:** Findings reflect the evaluated dataset, partitioning, and model configurations; cross-vehicle and cross-OEM generalization has not been tested.
6. **Desktop Benchmarking Scope:** Latency was profiled on an x86_64 desktop CPU in Python/PyTorch ($B=1$) to establish relative algorithmic complexity; it does not constitute embedded automotive microcontroller (ECU) hardware certification or hard real-time safety compliance.

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/kritik8/Federated-IDS-CAN.git
cd Federated-IDS-CAN

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Test Suite
The repository includes a comprehensive unit, integration, and regression invariant test suite:
```bash
python -m pytest tests
```
*Expected Output: `53 passed in ~45s` with 0 failures and 0 warnings.*

### 3. Inspect Pre-Executed Notebooks & Results
- `notebooks/01_dataset_analysis.ipynb`: Raw CAN exploratory data analysis.
- `notebooks/02_preprocessing.ipynb`: Sliding window generation and chronological splits.
- `notebooks/03_baseline_ids.ipynb`: Centralized 1D-CNN baseline execution.
- `notebooks/04_federated_iid.ipynb`: Federated IID baseline evaluation.
- `notebooks/05_federated_noniid.ipynb`: Non-IID Dirichlet heterogeneity benchmarks.
- `notebooks/06_robust_federated_learning.ipynb`: Byzantine attacks and robust defenses.

### 4. Authoritative Metrics and Reports
- Master verified results table: [`results/metrics/final_verified_results.csv`](results/metrics/final_verified_results.csv)
- Latency and speedup reconciliation: [`final_audit_2026/LATENCY_SPEEDUP_RECONCILIATION.md`](final_audit_2026/LATENCY_SPEEDUP_RECONCILIATION.md)
- Non-IID result reconciliation: [`final_audit_2026/NONIID_RESULT_RECONCILIATION.md`](final_audit_2026/NONIID_RESULT_RECONCILIATION.md)
- Implementation freeze report: [`final_audit_2026/FINAL_IMPLEMENTATION_FREEZE_REPORT.md`](final_audit_2026/FINAL_IMPLEMENTATION_FREEZE_REPORT.md)
- Post-freeze verification report: [`POST_FREEZE_VERIFICATION_REPORT.md`](POST_FREEZE_VERIFICATION_REPORT.md)

---

## 📁 Repository Structure

```text
Federated-IDS-CAN/
│
├── configs/                                # Experiment configurations
│   ├── random_forest_config.json           # 300-estimator Random Forest baseline config
│   └── robust_federated_config.json        # Threat models and robust aggregator settings
│
├── src/                                    # Core implementation modules
│   ├── data/                               # Dynamic variable DLC CSV loader
│   ├── preprocessing/                      # 11-feature parser, windowing, and chronological splitter
│   ├── models/                             # CAN1DCNN with GroupNorm (G=4)
│   ├── evaluation/                         # Metric calculations, confusion matrices, latency profiler
│   └── federated/                          # Client SGD, Server routing, Threat models, Aggregation rules
│
├── tests/                                  # Automated test suite (53 tests passing)
│   ├── test_loader.py                      # CSV parsing and variable DLC verification
│   ├── test_models.py                      # Tensor shape propagation and GroupNorm invariant
│   ├── test_preprocessing.py               # Feature normalization and sliding window bounds
│   ├── test_metrics.py                     # Metric calculation edge cases and warning suppression
│   ├── test_federated.py                   # Client training and server communication loops
│   ├── test_robust_aggregation.py          # Median, Trimmed Mean, and threat model tests
│   ├── test_multi_krum.py                  # Multi-Krum formulation, bounds, and dispatching
│   └── test_validation_invariants.py       # Multi-seed ddof=1, Non-IID integrity, speedup ratios
│
├── results/                                # Experimental outputs and artifacts
│   ├── baseline_multiseed/                 # Centralized 1D-CNN runs (Seeds 42, 43, 44)
│   ├── iid_multiseed/                      # Federated IID runs (Seeds 42, 43, 44)
│   ├── noniid_multiseed/                   # Non-IID Dirichlet runs (α in {1.0, 0.5, 0.1})
│   ├── metrics/                            # Authoritative final CSV and JSON benchmark summaries
│   │   ├── final_verified_results.csv      # Master comparison table across all 17 configurations
│   │   ├── final_verified_results.json     # Machine-readable authoritative metrics
│   │   └── latency_and_size_audit.json     # Hardware latency and model size benchmark logs
│   └── figures/                            # Publication-ready visualizations (300 DPI)
│       ├── system_architecture.png         # Framework pipeline and threat model diagram
│       ├── centralized_cnn_vs_rf_comparison.png # 1D-CNN vs. Random Forest comparative bar chart
│       ├── multikrum_vs_robust_aggregators_f1.png # Grouped F1 comparison under poisoning
│       └── robust_per_attack_recall_attack_b.png # Per-attack recall breakdown under Attack B
│
├── final_audit_2026/                       # Independent scientific audit and reconciliation reports
│   ├── FINAL_IMPLEMENTATION_FREEZE_REPORT.md # Official implementation freeze document
│   ├── NONIID_RESULT_RECONCILIATION.md     # Forensic analysis of Non-IID performance
│   ├── LATENCY_SPEEDUP_RECONCILIATION.md   # Mathematical derivation of latency speedups
│   ├── FINAL_FREEZE_MANIFEST.md            # Complete file and artifact inventory
│   └── POST_FREEZE_VERIFICATION_REPORT.md  # Read-only audit and errata verification
│
├── notebooks/                              # Pre-executed Jupyter research notebooks (01 to 06)
├── README.md                               # This documentation
└── requirements.txt                        # Environment dependency specification
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE). If you utilize this implementation or benchmark results in academic research, please cite:

```bibtex
@misc{federated_ids_can_2026,
  title={Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks},
  author={Jain, Kritik},
  year={2026},
  howpublished={\url{https://github.com/kritik8/Federated-IDS-CAN}}
}
```
