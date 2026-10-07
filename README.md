<div align="center">

# Federated-IDS-CAN

**Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_1_Baseline_Hardened-brightgreen?style=flat-square)]()

<p align="center">
  A research-grade, scientifically reproducible intrusion detection system (IDS) baseline and upcoming decentralized Federated Learning framework designed for heterogeneous in-vehicle Controller Area Network (CAN) security.
</p>

[Project](#-project) • [Why It Matters](#-why-this-research-matters) • [Methodology](#-current-methodology) • [Dataset](#-current-dataset) • [Baseline Architecture](#-current-baseline-model) • [Results](#-current-results) • [Design Decisions](#-important-methodological-decisions) • [Limitations](#-current-limitations) • [Next Steps](#-next-step)

</div>

---

## 🚀 Project

**What are we building?**

We are developing a privacy-preserving, Byzantine-robust **Federated Learning (FL) Intrusion Detection System** for modern connected vehicles. 

In modern automobiles, dozens of Electronic Control Units (ECUs) communicate over the internal **Controller Area Network (CAN)** bus. Because CAN has no built-in encryption or authentication, attackers who breach infotainment, telematics, or OBD-II ports can inject malicious messages directly into steering, braking, and powertrain networks.

Instead of pooling sensitive telemetry to a central cloud server, our framework enables individual vehicles (or ECUs) to train local deep-learning intrusion detectors collaboratively on their own traffic traces, sharing only model weight updates.

---

## 🛡️ Why This Research Matters

1. **Zero Native CAN Security:** CAN bus broadcasts all messages in plaintext. Any compromised node can spoof packets with high arbitration priority.
2. **Safety-Critical Deadlines:** In-vehicle intrusions must be detected in sub-millisecond timeframes to prevent catastrophic physical manipulation.
3. **Data Privacy & Telemetry Volume:** Streaming raw CAN traffic from millions of vehicles to a centralized server causes severe bandwidth exhaustion and exposes private user location and driving patterns.
4. **Fleet Heterogeneity (Non-IID Traffic):** Different vehicle makes, firmware versions, and driving profiles generate non-identically distributed (Non-IID) traffic patterns that cause standard distributed algorithms (like vanilla FedAvg) to diverge.
5. **Adversarial Threats:** Compromised or malicious vehicles can launch Byzantine poisoning attacks to sabotage global detection models.

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
                                     ├──► [NEXT] Federated Learning simulation (FedAvg, FedProx, SCAFFOLD)
                                     ├──► [NEXT] Non-IID Dirichlet client partitioning
                                     ├──► [NEXT] Malicious clients & Byzantine attacks
                                     ├──► [NEXT] Robust aggregation (Median, Trimmed Mean, Krum, FoolsGold)
                                     └──► [NEXT] Hardware trace-replay evaluation testbed
```

| Phase | Milestone | Status | Description |
|---|---|:---:|---|
| **Phase 1** | Centralized Baseline Hardening | **DONE** | Fixed variable DLC CSV parsing, converted to GroupNorm, eliminated inter-arrival leakage, bounded $\Delta t$ into $[0, 1]$, enabled best validation checkpointing, added microsecond edge latency benchmarking. |
| **Phase 1** | Full Notebook Execution | **DONE** | Executed and verified `notebooks/01_dataset_analysis.ipynb`, `02_preprocessing.ipynb`, and `03_baseline_ids.ipynb`. |
| **Phase 2** | Federated Learning Harness | **NEXT** | Simulate $K=10$ vehicular ECUs, Dirichlet Non-IID partitioning, FedAvg / FedProx / SCAFFOLD baselines. |
| **Phase 3** | Byzantine Defense Layer | **NEXT** | Poisoning attacks (label flipping, model poisoning) and robust aggregation rules (Krum, Coordinate-wise Median, FoolsGold). |
| **Phase 4** | Trace Replay & Paper Benchmarks | **NEXT** | Empirical evaluation on streaming CAN hardware traces. |

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

We use a lightweight, specialized **1D Convolutional Neural Network (`CAN1DCNN`)** designed specifically for edge vehicular ECUs:

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
- **Parameters:** 40,610 trainable parameters (~162.4 KB state dict).
- **Normalization:** `GroupNorm` with 4 groups per layer (replaces `BatchNorm1d` to ensure stability under future Non-IID client distributions).

---

## 📊 Current Results

Results produced by executing the hardened centralized baseline (checkpoint selected at **Epoch 8** with best validation F1 = 99.42%) on the untouched chronological test set (4,690 windows: 1,790 Attack, 2,900 Normal):

| Metric | Measured Value | Metric | Measured Value |
|---|---|---|---|
| **Test Accuracy** | **99.66%** | **F1-Score** | **99.55%** |
| **Precision** | **99.61%** | **ROC-AUC** | **0.9992** |
| **Recall (Sensitivity)** | **99.50%** | **False Positive Rate (FPR)** | **0.24%** (7 / 2,900) |
| **True Positives (TP)** | **1,781** | **False Negative Rate (FNR)** | **0.50%** (9 / 1,790) |
| **True Negatives (TN)** | **2,893** | **Training Time (10 epochs)** | **27.93 seconds** (CPU) |

### Inference Latency & Resource Profiling
Evaluated separately to distinguish real-time single-window response from batched pipeline throughput:

- **Single-Sample Latency (`batch_size = 1`):**
  - Mean: **455.99 $\mu$s (0.46 ms)**
  - Median: **421.05 $\mu$s (0.42 ms)**
  - 95th Percentile: **590.59 $\mu$s (0.59 ms)**
  *(Strictly complies with the $<1.0$ ms real-time ECU CAN frame deadline).*
- **Batched Execution (`batch_size = 128`):**
  - Batch Latency: **3.46 ms $\pm$ 0.31 ms**
  - Amortized Per-Sample: **27.04 $\mu$s**
  - Ingestion Throughput: **36,975 samples/sec**

### Reproducibility Verification
Running the entire centralized pipeline from scratch using `seed = 42` produces **identical bit-for-bit metrics** (0.000000% difference across Accuracy, Precision, Recall, F1, FPR, FNR, ROC-AUC, and Confusion Matrix).

---

## ⚖️ Important Methodological Decisions

1. **Chronological Per-Scenario Splitting:** Random shuffling destroys the temporal autocorrelation of packet streams and causes severe message leakage. We preserve strict chronological order (70% train, 15% val, 15% test) within each capture scenario before aggregation.
2. **16-Frame Sliding Windows ($S=W=16$):** Individual CAN frames carry minimal contextual payload; grouping into 16 consecutive frames captures frequency spikes and sequential bit transitions without cross-boundary frame overlap.
3. **GroupNorm over BatchNorm:** `BatchNorm1d` computes running statistics per batch. In Non-IID Federated Learning, client running statistics drift apart, degrading the aggregated model. `GroupNorm` normalizes per sample across 4 channel groups, eliminating weight-drift.
4. **Domain-Based Normalization:** Normalizing CAN ID (by 2047.0), DLC (by 8.0), payload bytes (by 255.0), and $\log_{10}(1 + \Delta t \times 1000)$ (by $\log_{10}(1001.0)$) uses protocol constants rather than training dataset sample statistics, preventing data leakage and feature scale dominance.
5. **Leak-Free Initial Inter-Arrival Time:** Frame 0 inter-arrival time is initialized to 0.0 (reflecting no prior observed packet at startup), replacing previous global median calculations that leaked future test timestamps.
6. **Best Validation Checkpoint Selection:** Checkpoints are selected based strictly on validation F1 score during training, keeping the test set untouched until final evaluation.

---

## ⚠️ Current Limitations

- **Dataset Subsampling:** The benchmark currently ingests the first 100,000 frames per scenario (500,000 total frames out of 17.5M available in HCRL).
- **Simulated Federation:** The current phase validates the centralized detector; client partitioning across distinct vehicle ECUs is simulated in Phase 2 rather than recorded from multiple physically distinct vehicles.
- **Trace-Driven Replay vs Full Digital Twin:** This framework evaluates trace-driven hardware replay; it is not yet a complete closed-loop cyber-physical vehicle dynamics simulator (e.g., CARLA or SUMO co-simulation).

---

## 🧭 Next Step

```text
NEXT:
Build the Federated Learning simulation, beginning with client partitioning and the IID baseline.
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
│   └── evaluation/          # metrics.py (metrics computation, latency profiling)
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb   # Executed EDA, CAN ID & class distributions
│   ├── 02_preprocessing.ipynb      # Executed feature extraction & splitting
│   └── 03_baseline_ids.ipynb       # Executed baseline training & evaluation
│
├── results/
│   ├── figures/             # Confusion matrices, training curves, EDA figures
│   ├── metrics/             # centralized_baseline_v2.json, baseline_results.json
│   ├── models/              # centralized_1d_cnn_best.pt, baseline_1d_cnn.pt
│   └── reports/             # centralized_baseline_v2.md (audit & evaluation report)
│
├── configs/                 # baseline_config.json
├── tests/                   # test_loader.py, test_models.py, test_preprocessing.py
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

### 2. Run Test Suite
```bash
python -m unittest discover tests
```

### 3. Run Pipeline via Notebooks
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_ids.ipynb
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
