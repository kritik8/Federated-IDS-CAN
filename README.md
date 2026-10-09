<div align="center">

# Federated-IDS-CAN

**Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_2A_IID_Federated_Complete-brightgreen?style=flat-square)]()

<p align="center">
  A research-grade, scientifically reproducible intrusion detection system (IDS) baseline and decentralized Federated Learning framework designed for heterogeneous in-vehicle Controller Area Network (CAN) security.
</p>

[Project](#-project) • [Why It Matters](#-why-this-research-matters) • [Methodology](#-current-methodology) • [Dataset](#-current-dataset) • [Baseline Architecture](#-current-baseline-model) • [Phase 2A Federated Baseline](#-phase-2a-iid-federated-learning-baseline) • [Results & Comparison](#-results--centralized-vs-federated-comparison) • [Design Decisions](#-important-methodological-decisions) • [Limitations](#-current-limitations) • [Next Steps](#-next-step)

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
                                     ├──► [DONE] Phase 2A: 10-Client IID FedAvg baseline
                                     ├──► [NEXT] Phase 2B: Non-IID Dirichlet client partitioning
                                     ├──► [NEXT] Phase 2C: Heterogeneity-aware FL (FedProx, SCAFFOLD)
                                     ├──► [NEXT] Phase 3: Malicious clients & Byzantine attacks
                                     ├──► [NEXT] Phase 3B: Robust aggregation (Median, Trimmed Mean, Krum, FoolsGold)
                                     └──► [NEXT] Phase 4: Hardware trace-replay evaluation testbed
```

| Phase | Milestone | Status | Description |
|---|---|:---:|---|
| **Phase 1** | Centralized Baseline Hardening | **DONE** | Fixed variable DLC CSV parsing, converted to GroupNorm, eliminated inter-arrival leakage, bounded $\Delta t$ into $[0, 1]$, enabled best validation checkpointing, added microsecond edge latency benchmarking. |
| **Phase 1** | Centralized Notebook Execution | **DONE** | Executed and verified `notebooks/01_dataset_analysis.ipynb`, `02_preprocessing.ipynb`, and `03_baseline_ids.ipynb`. |
| **Phase 2A** | **IID Federated Baseline** | **DONE** | Simulated 10 vehicular clients under deterministic stratified IID partitioning, implemented sample-weighted FedAvg, achieved **98.25% Accuracy** and **97.66% F1** across 10 rounds. Executed `notebooks/04_federated_iid.ipynb`. |
| **Phase 2B** | Non-IID Dirichlet Partitioning | **NEXT** | Partition client datasets under Dirichlet distributions ($\alpha \in \{0.1, 0.5, 1.0\}$) to measure client drift and performance degradation under fleet heterogeneity. |
| **Phase 2C** | Heterogeneity-Aware Algorithms | **PLANNED** | Implement FedProx ($\mu$-proximal regularization) and SCAFFOLD (control variates) to counter Non-IID drift. |
| **Phase 3** | Byzantine Attacks & Robust Aggregation | **PLANNED** | Implement label flipping, additive Gaussian noise, and targeted backdoor attacks; evaluate Coordinate-wise Median, Trimmed Mean, Krum, and FoolsGold. |
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
- **Parameters:** 40,610 trainable parameters (~158.6 KB state dict).
- **Normalization:** `GroupNorm` with 4 groups per layer (replaces `BatchNorm1d` to eliminate running mean/variance divergence in distributed federated settings).

---

## 🌐 Phase 2A: IID Federated Learning Baseline

### What is Federated Learning?
In traditional machine learning, all data must be transmitted to a central server. In Federated Learning (FL), **training is decentralized**:
1. The central server distributes a global model to participating clients (vehicles or ECUs).
2. Each client trains the model on its private local data without ever transmitting raw CAN frames.
3. Each client sends only updated model parameters (weights and biases) back to the server.
4. The server aggregates these updates into a new global model using **Federated Averaging (FedAvg)**.

### What is a "Simulated Client"?
In our benchmark, we instantiate **10 simulated clients** representing distinct connected vehicles or domain gateway ECUs. Each client maintains an isolated local training dataset and its own instance of `CAN1DCNN`. It never observes another client's samples.

### What does "IID" Mean?
**IID** (*Independent and Identically Distributed*) means:
- **Identically Distributed:** Each of the 10 clients has approximately the exact same class balance (~60.9% normal traffic, ~39.1% cyber-attacks) as the global dataset.
- **Independent:** Each client's data partition is a disjoint, non-overlapping subset of the centralized training pool.
- **Why start with IID?** The IID setup provides an ideal, noise-free collaborative benchmark. It establishes the performance ceiling before introducing the challenges of real-world fleet heterogeneity (Non-IID distributions) and adversarial attacks.

### New Federated Modules (`src/federated/`)
- [`src/federated/dataset.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/dataset.py): Implements deterministic stratified IID partitioning among $K=10$ clients. Ensures 0 duplicate indices, 0 omissions, and balanced class distributions.
- [`src/federated/aggregation.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/aggregation.py): Implements sample-weighted FedAvg ($\theta_{\text{global}} = \sum_k \frac{n_k}{N} \theta_k$) with strict tensor compatibility, dtype, and GroupNorm floating-point parameter checks. Provides analytical communication cost estimation.
- [`src/federated/client.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/client.py): `FederatedClient` class managing local model parameter updates, local Adam training, detached CPU weight extraction, and training loss diagnostics.
- [`src/federated/server.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/src/federated/server.py): `FederatedServer` orchestrator handling parameter broadcast, client execution, FedAvg aggregation, round-by-round validation tracking, best checkpoint restoration based on Validation F1, and final test evaluation.

### Client Sample Distribution (10 Clients, Seed 42):

| Client ID | Total Samples | Normal ($y=0$) | Attack ($y=1$) | Attack % | DoS | Fuzzy | Gear | RPM |
|---|---|---|---|---|---|---|---|---|
| **Client 0** | 2,189 | 1,333 | 856 | 39.10% | 171 | 167 | 264 | 254 |
| **Client 1** | 2,189 | 1,333 | 856 | 39.10% | 207 | 176 | 238 | 235 |
| **Client 2** | 2,188 | 1,332 | 856 | 39.12% | 181 | 149 | 266 | 260 |
| **Client 3** | 2,187 | 1,332 | 855 | 39.09% | 168 | 150 | 273 | 264 |
| **Client 4** | 2,187 | 1,332 | 855 | 39.09% | 194 | 161 | 252 | 248 |
| **Client 5** | 2,187 | 1,332 | 855 | 39.09% | 212 | 146 | 222 | 275 |
| **Client 6** | 2,187 | 1,332 | 855 | 39.09% | 205 | 164 | 244 | 242 |
| **Client 7** | 2,187 | 1,332 | 855 | 39.09% | 198 | 155 | 255 | 247 |
| **Client 8** | 2,187 | 1,332 | 855 | 39.09% | 185 | 160 | 243 | 267 |
| **Client 9** | 2,187 | 1,332 | 855 | 39.09% | 199 | 162 | 251 | 243 |
| **Total Pool** | **21,875** | **13,322** | **8,553** | **39.10%** | **1,920** | **1,590** | **2,508** | **2,535** |

---

## 📊 Results & Centralized vs. Federated Comparison

The federated model was trained for 10 rounds ($E=1$ local epoch per round, Adam, $\eta=0.001$, batch size 128). The best checkpoint was selected at **Round 9** (Validation F1 = 97.99%) and evaluated on the untouched test partition (4,690 windows: 2,900 Normal, 1,790 Attack):

| Metric | Centralized 1D-CNN (Phase 1) | Federated IID FedAvg (Phase 2A) | Delta | Notes |
|---|---|---|---|---|
| **Test Accuracy** | **99.66%** | **98.25%** | -1.41% | High accuracy retained without centralized pooling |
| **Test Precision** | 99.61% | **99.71%** | **+0.10%** | Federated model yields even fewer false alarms |
| **Test Recall** | **99.50%** | **95.70%** | -3.80% | 1,713 / 1,790 attack windows successfully detected |
| **Test F1-Score** | **99.55%** | **97.66%** | -1.89% | Near-centralized benchmark performance |
| **False Positive Rate (FPR)** | 0.24% | **0.17%** | **-0.07%** | Only 5 false alarms out of 2,900 normal windows |
| **False Negative Rate (FNR)** | **0.50%** | **4.30%** | +3.80% | 77 missed attack windows (40 in subtle Fuzzy attacks) |
| **ROC-AUC** | **0.9992** | **0.9965** | -0.0027 | Outstanding class separation |
| **Training Budget** | 10 epochs | 10 rounds $\times$ 1 epoch | Equal Volume | Identical total sample passes (21,875 samples/epoch) |
| **Training Wall Time** | 27.93s | 35.45s | +7.52s | Fast execution on local CPU (~3.5s per round) |
| **Privacy Preservation** | ❌ Raw CAN shared | ✅ Zero raw data shared | **Guaranteed** | No telemetry or location traces leave the client |

### Communication Footprint (Analytic Estimation):
- **Model Parameters:** 40,610 float32 parameters
- **Per-Client State Dict Size:** **158.63 KB** (162,440 bytes)
- **Round Downlink (Server $\to$ 10 Clients):** 1.55 MB (1,586.3 KB)
- **Round Uplink (10 Clients $\to$ Server):** 1.55 MB (1,586.3 KB)
- **Total Network Traffic per Round:** **3.10 MB** (3,172.7 KB)
*(Calculated from serialized float32 state dict sizes; not physically measured across network sockets).*

---

## ⚖️ Important Methodological Decisions

1. **Chronological Per-Scenario Splitting:** Random shuffling destroys the temporal autocorrelation of packet streams and causes severe message leakage. We preserve strict chronological order (70% train, 15% val, 15% test) within each capture scenario before aggregation.
2. **16-Frame Sliding Windows ($S=W=16$):** Individual CAN frames carry minimal contextual payload; grouping into 16 consecutive frames captures frequency spikes and sequential bit transitions without cross-boundary frame overlap.
3. **GroupNorm over BatchNorm:** `BatchNorm1d` computes running statistics per batch. In Non-IID Federated Learning, client running statistics drift apart, degrading the aggregated model. `GroupNorm` normalizes per sample across 4 channel groups, eliminating weight-drift.
4. **Domain-Based Normalization:** Normalizing CAN ID (by 2047.0), DLC (by 8.0), payload bytes (by 255.0), and $\log_{10}(1 + \Delta t \times 1000)$ (by $\log_{10}(1001.0)$) uses protocol constants rather than training dataset sample statistics, preventing data leakage and feature scale dominance.
5. **Leak-Free Initial Inter-Arrival Time:** Frame 0 inter-arrival time is initialized to 0.0 (reflecting no prior observed packet at startup), replacing previous global median calculations that leaked future test timestamps.
6. **Best Validation Checkpoint Selection:** Checkpoints are selected based strictly on validation F1 score during training, keeping the test set untouched until final evaluation.
7. **Strict State Dict Averaging:** FedAvg validates parameter dictionary shapes, keys, and floating-point types before aggregation, preventing silent corruption of model weights.

---

## ⚠️ Current Limitations

- **Simulated Federation:** The 10 clients are simulated partitions of the single Kia Soul dataset, rather than 10 physically distinct physical vehicles with diverse ECU architectures.
- **IID Distribution Assumption:** Traffic is currently distributed identically across all 10 clients. In real-world vehicular fleets, different vehicles experience vastly different environments and driving styles (Non-IID traffic).
- **Honest Clients Only:** All 10 clients are cooperative and benign. No Byzantine noise, poisoned updates, or label flipping attacks have been evaluated yet.
- **Trace-Driven Replay vs Full Digital Twin:** This framework evaluates trace-driven hardware replay; it is not yet a closed-loop cyber-physical vehicle dynamics simulator.

---

## 🧭 Next Step

```text
NEXT:
Phase 2B — Introduce controlled Non-IID client distributions via Dirichlet partitioning (alpha in {0.1, 0.5, 1.0})
and measure how statistical heterogeneity affects FedAvg convergence and detection accuracy.
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
│   └── 04_federated_iid.ipynb       # Executed Phase 2A IID Federated baseline (10 clients, FedAvg)
│
├── results/
│   ├── figures/             # Confusion matrices, training curves, EDA figures
│   ├── metrics/             # centralized_baseline_v2.json, federated_iid_fedavg.json
│   ├── models/              # centralized_1d_cnn_best.pt, federated_iid_fedavg_best.pt
│   └── reports/             # centralized_baseline_v2.md, federated_iid_report.md
│
├── configs/                 # baseline_config.json, federated_iid_config.json
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

### 2. Run Test Suite (18 Unit Tests)
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
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
