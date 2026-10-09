<div align="center">

# Federated-IDS-CAN

**Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_2B_Non--IID_Complete-brightgreen?style=flat-square)]()

<p align="center">
  A research-grade, scientifically reproducible intrusion detection system (IDS) benchmark evaluating decentralized Federated Learning under fleet heterogeneity (Non-IID traffic) on Controller Area Network (CAN) security.
</p>

[Project](#-project) • [Why It Matters](#-why-this-research-matters) • [Methodology](#-current-methodology) • [Dataset](#-current-dataset) • [Baseline Architecture](#-current-baseline-model) • [Phase 2A IID Baseline](#-phase-2a-iid-federated-learning-baseline) • [Phase 2B Non-IID Experiments](#-phase-2b-non-iid-heterogeneity-experiments) • [Results & Comparison](#-results--centralized-vs-iid-vs-non-iid-comparison) • [Methodological Decisions](#-important-methodological-decisions) • [Limitations](#-current-limitations) • [Next Steps](#-next-step)

</div>

---

## 🚀 Project

**What are we building?**

We are developing a Byzantine-robust **Federated Learning (FL) Intrusion Detection System** for modern connected vehicles. 

In modern automobiles, dozens of Electronic Control Units (ECUs) communicate over the internal **Controller Area Network (CAN)** bus. Because CAN has no built-in encryption or authentication, attackers who breach infotainment, telematics, or OBD-II ports can inject malicious messages directly into safety-critical networks.

Instead of pooling gigabytes of raw CAN telemetry into a centralized cloud server, our framework enables individual vehicles (or ECUs) to train local deep-learning intrusion detectors collaboratively on their own traffic traces, sharing only model weight updates.

> **Important Note on Privacy:** Local decentralized training ensures raw CAN data remains on each client. However, parameter sharing alone does *not* constitute a formal cryptographic or differential privacy guarantee against gradient leakage or model inversion attacks.

---

## 🛡️ Why This Research Matters

1. **Zero Native CAN Security:** CAN bus broadcasts all messages in plaintext. Any compromised node can spoof packets with high arbitration priority.
2. **Safety-Critical Deadlines:** In-vehicle intrusions must be detected in sub-millisecond timeframes to prevent catastrophic physical manipulation.
3. **Bandwidth Constraints:** Streaming raw CAN traffic from millions of connected vehicles causes telemetry bottlenecks.
4. **Fleet Heterogeneity (Non-IID Traffic):** Real vehicles experience different driving environments, routes, and threat profiles. This non-identical data distribution (Non-IID) induces **client drift**, causing standard distributed algorithms (like vanilla FedAvg) to degrade.
5. **Adversarial Threat Vectors:** In open vehicular fleets, compromised vehicles can launch Byzantine poisoning attacks to sabotage global detection.

---

## 🔬 Current Methodology

The overall research pipeline and its current execution status:

```text
CAN raw data
  └──► [DONE] Preprocessing & Variable DLC parsing
         └──► [DONE] 11-Dimensional Feature extraction
                └──► [DONE] Non-overlapping sliding windowing (W=16, S=16)
                       └──► [DONE] Leak-free chronological train/val/test splitting
                              └──► [DONE] Centralized 1D-CNN baseline (GroupNorm)
                                     ├──► [DONE] Phase 2A: 10-Client IID FedAvg baseline
                                     ├──► [DONE] Phase 2B: Non-IID Dirichlet client partitioning (alpha in {1.0, 0.5, 0.1})
                                     ├──► [NEXT] Phase 3: Malicious clients & Byzantine attacks (label-flipping, noise, backdoor)
                                     ├──► [NEXT] Phase 3B: Robust aggregation (Median, Trimmed Mean, Krum, FoolsGold)
                                     ├──► [PLANNED] Phase 3C: Heterogeneity-aware FL (FedProx, SCAFFOLD)
                                     └──► [PLANNED] Phase 4: Hardware trace-replay evaluation testbed
```

| Phase | Milestone | Status | Description |
|---|---|:---:|---|
| **Phase 1** | Centralized Baseline Hardening | **DONE** | Fixed variable DLC CSV parsing, converted to GroupNorm, eliminated inter-arrival leakage, bounded $\Delta t$ into $[0, 1]$, enabled best validation checkpointing, added edge latency profiling. |
| **Phase 1** | Centralized Notebook Execution | **DONE** | Executed and verified `notebooks/01_dataset_analysis.ipynb`, `02_preprocessing.ipynb`, and `03_baseline_ids.ipynb`. |
| **Phase 2A** | IID Federated Baseline | **DONE** | Simulated 10 vehicular clients under deterministic stratified IID partitioning, implemented sample-weighted FedAvg, achieved **98.25% Accuracy** and **97.66% F1**. Executed `notebooks/04_federated_iid.ipynb`. |
| **Phase 2B** | **Non-IID Heterogeneity Benchmark** | **DONE** | Partitioned training pool across 10 clients using Dirichlet distributions ($\alpha \in \{1.0, 0.5, 0.1\}$) based on attack categories. Quantified client drift and cyber-threat vulnerability under FedAvg. Executed `notebooks/05_federated_noniid.ipynb`. |
| **Phase 3** | Byzantine Attacks & Threat Model | **NEXT** | Formalize vehicular adversarial threat models: label-flipping, additive Gaussian noise, and targeted backdoor attacks. |
| **Phase 3B** | Robust Aggregation Rules | **PLANNED** | Implement and evaluate Byzantine-robust aggregators: Coordinate-wise Median, Trimmed Mean, Krum, and FoolsGold. |
| **Phase 4** | Trace Replay & Paper Benchmarks | **PLANNED** | Hardware trace-replay evaluation testbed on embedded edge platforms. |

---

## 💾 Current Dataset

We evaluate our pipeline on the widely recognized **HCRL Car-Hacking Dataset** recorded by the Korea University Hacking and Countermeasure Research Lab from a commercial Kia Soul vehicle:

- **Normal Run (`normal_run_data.txt`):** Ambient driving traffic without cyber-attacks.
- **DoS Attack (`DoS_dataset.csv`):** Injected dominant `0x000` messages at 0.3 ms frequency to overwhelm bus arbitration.
- **Fuzzy Attack (`Fuzzy_dataset.csv`):** Injected random arbitration IDs and randomized payloads across the bus.
- **RPM Spoofing (`RPM_dataset.csv`):** Target injection spoofing engine RPM sensor readings (`CAN ID: 0x0316`).
- **Gear Spoofing (`gear_dataset.csv`):** Target injection spoofing automatic transmission gear status (`CAN ID: 0x043F`).

### Processed Dataset Volume
The pipeline subsamples the first **100,000 frames** from each of the 5 scenarios (500,000 frames total). Frames are grouped into non-overlapping contiguous 16-frame windows ($W=16, S=16$) and split chronologically per scenario:

- **Train Set (70%):** 21,875 windows (8,553 Attack, 13,322 Normal)
- **Validation Set (15%):** 4,685 windows (1,798 Attack, 2,887 Normal)
- **Test Set (15%):** 4,690 windows (1,790 Attack, 2,900 Normal)
- **Total Windows:** 31,250 windows

---

## 🧠 Current Baseline Model

We use a lightweight **1D Convolutional Neural Network (`CAN1DCNN`)** designed specifically for edge vehicular ECUs:

```
Input: (Batch, 16 frames, 11 features) ──► Transpose to (Batch, 11 channels, 16 steps)
  │
  ├── Conv1D (11 ──► 32, k=3, p=1) + GroupNorm(4, 32) + ReLU + MaxPool(2)
  ├── Conv1D (32 ──► 64, k=3, p=1) + GroupNorm(4, 64) + ReLU + MaxPool(2)
  ├── Conv1D (64 ──► 128, k=3, p=1) + GroupNorm(4, 128) + ReLU + AdaptiveAvgPool(1)
  │
  └── Dropout(0.2) ──► Linear(128 ──► 64) ──► ReLU ──► Dropout(0.2) ──► Linear(64 ──► 2)
```

- **Feature Dimensions (11):** Normalized CAN ID ($[0, 1]$), normalized DLC ($[0, 1]$), 8 payload bytes ($D_0 \dots D_7 \in [0, 1]$), and domain-scaled log inter-arrival time ($\Delta t \in [0, 1]$).
- **Parameters:** 40,610 trainable parameters (~158.6 KB state dict).
- **Normalization:** `GroupNorm` with 4 groups per layer (replaces `BatchNorm1d` to eliminate running mean/variance divergence in distributed federated settings).

---

## 🌐 Phase 2A: IID Federated Learning Baseline

In Phase 2A, 10 simulated vehicular clients were allocated balanced, stratified IID subsets of the training pool (~60.9% Normal, ~39.1% Attack). Clients trained locally for $E=1$ local epoch per round, communicating parameter updates to a server executing sample-weighted Federated Averaging (FedAvg).

- **Result:** Achieved **98.25% Test Accuracy** and **97.66% F1-score** across 10 rounds, establishing the collaborative performance ceiling without centralizing raw CAN traces.

---

## 🌐 Phase 2B: Non-IID Heterogeneity Experiments

### What is Non-IID and Why Introduce It?
Real connected vehicles do not experience identical driving traffic. One vehicle may be subjected to an active Denial-of-Service or bus-flooding attack, while nine other vehicles cruise under benign normal conditions.

To rigorously model this fleet heterogeneity, we use the **Dirichlet distribution** $\text{Dir}(\alpha \cdot \mathbf{1}_K)$:
- **What $\alpha$ (Alpha) Represents:**
  - **$\alpha = 1.0$ (Moderate Skew):** Clients have unbalanced proportions of normal and attack traffic, but all clients have access to most traffic categories.
  - **$\alpha = 0.5$ (High Skew):** Pronounced heterogeneity. Several clients lack exposure to specific cyber-attack types.
  - **$\alpha = 0.1$ (Extreme Skew):** Severe specialization. Multiple clients observe 100% cyber-attack traffic with 0 normal samples, while other clients observe >90% normal traffic.
- **Attack-Type-Aware Partitioning:** The 5 underlying training categories (Normal, DoS, Fuzzy, Gear, RPM) are partitioned via Dirichlet allocation and mapped back to binary labels ($y \in \{0, 1\}$). This creates realistic attack-exposure diversity while preserving the binary IDS task.
- **Feasibility Constraint:** Rejection sampling ensures every client has at least 64 samples to enable valid mini-batch gradient descent.

---

## 📊 Results: Centralized vs. IID vs. Non-IID Comparison

All models were evaluated on the untouched test partition (4,690 windows: 2,900 Normal, 1,790 Attack) using the best checkpoint selected by Validation F1:

| Benchmark Regime | Best Round | Val F1 (%) | Test Acc (%) | Test Prec (%) | Test Recall (%) | Test F1 (%) | FPR (%) | FNR (%) | ROC-AUC | Wall Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | Ep 8 | 99.42% | **99.66%** | 99.61% | **99.50%** | **99.55%** | 0.24% | **0.50%** | **0.9992** | 25.12s |
| **Federated IID (FedAvg)** | Rd 9 | 97.99% | **98.25%** | 99.71% | 95.70% | **97.66%** | 0.17% | 4.30% | 0.9965 | 35.45s |
| **Non-IID ($\alpha = 1.0$)** | Rd 10 | 98.05% | **98.51%** | **99.83%** | 96.26% | **98.01%** | **0.10%** | 3.74% | 0.9961 | 28.73s |
| **Non-IID ($\alpha = 0.5$)** | Rd 9 | 97.94% | **98.49%** | 99.60% | 96.42% | **97.98%** | 0.24% | 3.58% | 0.9956 | 25.51s |
| **Non-IID ($\alpha = 0.1$, Extreme)** | Rd 10 | 90.12% | **92.11%** | 97.91% | **81.06%** | **88.69%** | **1.07%** | **18.94%** | **0.9573** | 27.41s |

### Per-Attack-Type Detection Rate Analysis

| Cyber-Attack Class | Total Test Windows | IID Baseline | Non-IID $\alpha=1.0$ | Non-IID $\alpha=0.5$ | Non-IID $\alpha=0.1$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Normal Traffic** | 2,900 | 99.83% | 99.90% | 99.76% | **98.93%** |
| **Gear Spoofing** | 591 | 97.63% | 99.32% | 98.48% | **100.00%** |
| **RPM Spoofing** | 592 | 97.47% | 99.32% | 100.00% | **100.00%** |
| **DoS Injection** | 398 | 97.99% | 97.99% | 98.24% | **43.97% ⚠️** |
| **Fuzzy Injection** | 209 | 80.86% | 75.60% | 77.03% | **44.50% ⚠️** |

### Key Scientific Takeaways:
1. **Resilience to Moderate Non-IID:** For $\alpha \in \{1.0, 0.5\}$, FedAvg with `GroupNorm` maintains high detection accuracy ($98.0\%$ F1), showing that mild fleet heterogeneity does not break standard federated averaging.
2. **Catastrophic Drift under Extreme Non-IID ($\alpha = 0.1$):** When clients experience complete label separation (some clients observing only attacks, others observing mostly normal traffic), FedAvg suffers severe gradient conflict. False negative rate jumps from $4.30\%$ to **$18.94\%$**, causing high-volume DoS and subtle Fuzzy injection attacks to be missed over half the time.
3. **Spoofing Signature Stability:** Semantic sensor spoofing (Gear, RPM) was detected at **100% accuracy** even under extreme heterogeneity due to strong fixed arbitration and bit patterns.

---

## ⚖️ Important Methodological Decisions

1. **Chronological Per-Scenario Splitting:** Random shuffling destroys temporal packet dynamics and leaks messages. We enforce strict chronological ordering (70% train, 15% val, 15% test).
2. **16-Frame Sliding Windows ($S=W=16$):** Grouping into 16 consecutive frames captures frequency bursts without frame overlap.
3. **GroupNorm over BatchNorm:** `GroupNorm` normalizes per sample across 4 channel groups, avoiding the catastrophic running statistics divergence that affects `BatchNorm1d` under Non-IID data.
4. **Domain-Based Normalization:** Protocol constants (CAN ID by 2047.0, DLC by 8.0, payload by 255.0, inter-arrival time by physical maximum) prevent dataset leakage.
5. **Leak-Free Initial Inter-Arrival Time:** Frame 0 inter-arrival time is initialized to 0.0, reflecting monitoring inception without future timestamp consumption.
6. **Best Validation Checkpoint Selection:** Selected strictly by validation F1 score, keeping the test set untouched until post-training evaluation.
7. **Strict Parameter Compatibility:** FedAvg validates parameter tensor keys, shapes, and floating-point types before aggregation.

---

## ⚠️ Current Limitations

- **Simulated Federation:** The 10 clients are simulated partitions of a single Kia Soul capture dataset, rather than 10 physically distinct vehicles with varied ECU architectures.
- **Benign Clients Only:** All clients in Phase 2A and 2B are honest and cooperative. No adversarial poisoning (label flipping, noise injection, backdoors) was present.
- **No Client Drift Regularization:** Vanilla FedAvg was used without proximal regularization (FedProx) or control variates (SCAFFOLD), which will be evaluated in subsequent phases.

---

## 🧭 Next Step

```text
NEXT:
Phase 3 — Introduce a formalized vehicular Byzantine threat model (label-flipping, additive Gaussian noise, 
and targeted backdoor attacks) and evaluate Byzantine-robust aggregation rules (Median, Trimmed Mean, Krum, FoolsGold).
```

---

## 📁 Repository Structure

```
Federated-IDS-CAN/
│
├── data/
│   ├── raw/                 # Raw HCRL captures (DoS, Fuzzy, Gear, RPM, Normal) [.gitignore]
│   ├── processed/           # can_ids_processed_dataset.npz, metadata.json [.gitignore]
│   └── splits/              # train_split.npz, val_split.npz, test_split.npz [.gitignore]
│
├── src/
│   ├── data/                # loader.py (dynamic variable DLC CSV parsing)
│   ├── preprocessing/       # parser.py, pipeline.py (features, windowing, split)
│   ├── models/              # cnn1d.py (CAN1DCNN with GroupNorm)
│   ├── evaluation/          # metrics.py (metrics computation, latency profiling)
│   └── federated/           # dataset.py, client.py, server.py, aggregation.py
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb   # Executed EDA, CAN ID & class distributions
│   ├── 02_preprocessing.ipynb      # Executed feature extraction & splitting
│   ├── 03_baseline_ids.ipynb       # Executed centralized baseline training & evaluation
│   ├── 04_federated_iid.ipynb       # Executed Phase 2A IID Federated baseline (10 clients, FedAvg)
│   └── 05_federated_noniid.ipynb    # Executed Phase 2B Non-IID Dirichlet benchmark (alpha in {1.0, 0.5, 0.1})
│
├── results/
│   ├── figures/             # Confusion matrices, training curves, Non-IID comparison plots
│   ├── metrics/             # centralized_baseline_v2.json, federated_iid_fedavg.json, federated_noniid_alpha_*.json
│   ├── models/              # centralized_1d_cnn_best.pt, federated_iid_fedavg_best.pt, federated_noniid_alpha_*_best.pt
│   └── reports/             # centralized_baseline_v2.md, federated_iid_report.md, federated_noniid_report.md
│
├── configs/                 # baseline_config.json, federated_iid_config.json, federated_noniid_config.json
├── tests/                   # test_loader.py, test_models.py, test_preprocessing.py, test_federated.py
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── AUDIT_REPORT.md
```

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

### 2. Run Test Suite (25 Unit Tests)
```bash
python -m unittest discover tests
```

### 3. Run Pipeline via Notebooks
```bash
# Phase 1: Centralized Pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_ids.ipynb

# Phase 2A: Federated IID Baseline (10 Clients, FedAvg)
jupyter nbconvert --to notebook --execute --inplace notebooks/04_federated_iid.ipynb

# Phase 2B: Non-IID Dirichlet Experiments (alpha in {1.0, 0.5, 0.1})
jupyter nbconvert --to notebook --execute --inplace notebooks/05_federated_noniid.ipynb
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
