<div align="center">

# Federated-IDS-CAN

**Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Centralized_Baseline_Verified-brightgreen?style=flat-square)]()

<p align="center">
  A lightweight, reproducible deep learning baseline and decentralized Federated Learning framework designed to detect in-vehicle Controller Area Network (CAN) intrusions across heterogeneous automotive ECUs.
</p>

[Dataset Access](#-dataset-download) • [Architecture](#-architecture) • [Results](#-baseline-results) • [Quickstart](#-quickstart) • [Roadmap](#-research-roadmap)

</div>

---

## 📌 Overview

Modern in-vehicle electronic control units (ECUs) communicate over the Controller Area Network (CAN) bus without native encryption or authentication. This project provides an end-to-end framework to detect high-priority injection, payload spoofing, and fuzzing attacks using:

- **11-Dimensional Frame Representations**: Arbitration IDs, DLC, 8 payload bytes, and log-scaled inter-arrival dynamics ($\Delta t$).
- **Contiguous Sliding Windows ($W=16$)**: Temporal sequences capturing inter-packet frequency patterns.
- **Lightweight 1D-CNN (~40.6k parameters)**: Tailored for memory-constrained edge vehicular microcontrollers.
- **Decentralized Federated Learning**: Designed for Non-IID client heterogeneity and Byzantine defense across distributed vehicular fleets.

---

## 💾 Dataset Download

This benchmark is evaluated on the widely recognized **HCRL Car-Hacking Dataset** recorded from a commercial vehicle (Kia Soul) via the OBD-II port by the Korea University Hacking & Countermeasure Research Lab.

| Scenario | File | Description | Download Link |
|---|---|---|:---:|
| **Normal Run** | `normal_run_data.txt` | Ambient driving CAN bus traffic without cyber-attacks | [Download (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) |
| **DoS Attack** | `DoS_dataset.csv` | Dominant `0x000` message injection at 0.3 ms frequency | [Download (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) |
| **Fuzzy Attack** | `Fuzzy_dataset.csv` | Random CAN IDs & payloads across address space | [Download (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) |
| **RPM Spoofing** | `RPM_dataset.csv` | Malicious payload injection to engine RPM (`0x0316`) | [Download (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) |
| **Gear Spoofing** | `gear_dataset.csv` | Malicious payload injection to transmission gear (`0x043F`) | [Download (HCRL)](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) |

> **Mirror & Alternative Download:**  
> The complete dataset collection is also accessible via the [Kaggle Car-Hacking Dataset Mirror](https://www.kaggle.com/datasets/subhajournal/car-hacking-dataset-csv).

### Placement
Download and extract the files directly into the `data/raw/` directory:
```
data/
└── raw/
    ├── DoS_dataset.csv
    ├── Fuzzy_dataset.csv
    ├── RPM_dataset.csv
    ├── gear_dataset.csv
    └── normal_run_data.txt
```
*(All raw data files are excluded from Git version control via `.gitignore`.)*

---

## 🧠 Architecture

```
Input: (Batch, 16 steps, 11 features) ──► Transpose to (Batch, 11 channels, 16 steps)
  │
  ├── Conv1D (11 ──► 32, k=3, p=1) + BatchNorm + ReLU + MaxPool(2)
  ├── Conv1D (32 ──► 64, k=3, p=1) + BatchNorm + ReLU + MaxPool(2)
  ├── Conv1D (64 ──► 128, k=3, p=1) + BatchNorm + ReLU + AdaptiveAvgPool(1)
  │
  └── Dropout(0.2) ──► Linear(128 ──► 64) ──► ReLU ──► Dropout(0.2) ──► Linear(64 ──► 2)
```

- **Parameters:** 40,610 (~162 KB)
- **Leakage Prevention:** Chronological partitioning (70% Train, 15% Val, 15% Test) applied strictly per scenario run before aggregation.

---

## 📊 Baseline Results

Evaluated on the untouched test partition (4,690 sequential windows: 1,790 Attack, 2,900 Normal):

| Metric | Measured Value | Metric | Measured Value |
|---|---|---|---|
| **Test Accuracy** | **99.62%** | **F1-Score** | **99.50%** |
| **Precision** | **99.72%** | **ROC-AUC** | **0.9995** |
| **Recall** | **99.27%** | **False Positive Rate (FPR)** | **0.17%** (5 / 2,900) |
| **Batch Latency (128)** | **3.59 ms** | **False Negative Rate (FNR)** | **0.73%** (13 / 1,790) |

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/kritik8/Federated-IDS-CAN.git
cd Federated-IDS-CAN
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Pipeline
Execute notebooks sequentially:
```bash
# 1. Exploratory Data Analysis & CAN ID distributions
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_analysis.ipynb

# 2. Feature Extraction, Windowing & Chronological Splitting
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb

# 3. 1D-CNN Baseline Training & Metrics Evaluation
jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_ids.ipynb
```

Artifacts will be generated in:
- `results/models/`: Saved weights (`baseline_1d_cnn.pt`)
- `results/metrics/`: Evaluation summary (`baseline_results.json`)
- `results/figures/`: Confusion matrices and training curves

---

## 📁 Repository Structure

```
Federated-IDS-CAN/
├── data/
│   ├── raw/             # Raw HCRL CSV & TXT captures (.gitignore)
│   ├── processed/       # Processed metadata & dataset arrays (.gitignore)
│   └── splits/          # Partition splits (.gitignore)
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_baseline_ids.ipynb
├── src/
│   ├── data/            # Dataset discovery & robust stream loaders
│   ├── preprocessing/   # CAN ID, payload parser & sliding window pipeline
│   ├── models/          # CAN1DCNN PyTorch model definitions
│   └── evaluation/      # Classification metrics & latency profiling
├── results/
│   ├── figures/         # Training curves & confusion matrices
│   ├── metrics/         # Benchmark evaluation JSONs
│   └── models/          # Model checkpoints (.pt)
├── requirements.txt
├── AUDIT_REPORT.md      # Comprehensive methodology audit report
└── README.md
```

---

## 🛣️ Research Roadmap

1. [x] **Centralized 1D-CNN Baseline**: Validated with 99.50% F1-score and deterministic reproducibility.
2. [ ] **Heterogeneous Partitioning**: Simulate $K$ vehicular clients with Dirichlet Non-IID distributions ($\alpha \in \{0.1, 0.5, 1.0\}$).
3. [ ] **Federated Aggregation Baselines**: Implement FedAvg, FedProx, and SCAFFOLD.
4. [ ] **Byzantine Robustness**: Implement defenses against malicious model poisoning (Coordinate-wise Median, Trimmed Mean, Krum, FoolsGold).
5. [ ] **CAN Trace Replay Testbed**: Closed-loop evaluation on streaming CAN traffic.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
