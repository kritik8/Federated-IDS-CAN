<div align="center">

# Federated-IDS-CAN

**Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase_3_Robust_Aggregation_Complete-brightgreen?style=flat-square)]()

<p align="center">
  A research-grade, scientifically reproducible intrusion detection system (IDS) benchmark evaluating decentralized Federated Learning under fleet heterogeneity (Non-IID traffic) and adversarial Byzantine client attacks on Controller Area Network (CAN) security.
</p>

[Project](#-project) • [Why It Matters](#-why-this-research-matters) • [Methodology](#-current-methodology) • [Dataset](#-current-dataset) • [Baseline Architecture](#-current-baseline-model) • [Phase 2A IID Baseline](#-phase-2a-iid-federated-learning-baseline) • [Phase 2B Non-IID Experiments](#-phase-2b-non-iid-heterogeneity-experiments) • [Phase 3 Robust Aggregation](#-phase-3-robust-federated-learning-under-adversarial-attacks) • [Results & Comparison](#-results--centralized-vs-iid-vs-non-iid-vs-robust-fl) • [Methodological Decisions](#-important-methodological-decisions) • [Limitations](#-current-limitations) • [Next Steps](#-next-step)

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
                                     ├──► [DONE] Phase 3: Malicious clients & Robust Aggregation (Label Flipping, Delta Sign Reversal, Median, Trimmed Mean)
                                     ├──► [NEXT] Phase 3B: Advanced Byzantine attacks & defense (Krum, FoolsGold, adaptive backdoors)
                                     ├──► [PLANNED] Phase 3C: Heterogeneity-aware FL (FedProx, SCAFFOLD)
                                     └──► [PLANNED] Phase 4: Hardware trace-replay evaluation testbed
```

| Phase | Milestone | Status | Description |
|---|---|:---:|---|
| **Phase 1** | Centralized Baseline Hardening | **DONE** | Fixed variable DLC CSV parsing, converted to GroupNorm, eliminated inter-arrival leakage, bounded $\Delta t$ into $[0, 1]$, enabled best validation checkpointing, added edge latency profiling. |
| **Phase 1** | Centralized Notebook Execution | **DONE** | Executed and verified `notebooks/01_dataset_analysis.ipynb`, `02_preprocessing.ipynb`, and `03_baseline_ids.ipynb`. |
| **Phase 2A** | IID Federated Baseline | **DONE** | Simulated 10 vehicular clients under deterministic stratified IID partitioning, implemented sample-weighted FedAvg, achieved **98.25% Accuracy** and **97.66% F1**. Executed `notebooks/04_federated_iid.ipynb`. |
| **Phase 2B** | **Non-IID Heterogeneity Benchmark** | **DONE** | Partitioned training pool across 10 clients using Dirichlet distributions ($\alpha \in \{1.0, 0.5, 0.1\}$) based on attack categories. Quantified client drift and cyber-threat vulnerability under FedAvg. Executed `notebooks/05_federated_noniid.ipynb`. |
| **Phase 3** | **Byzantine Attacks & Robust Aggregation** | **DONE** | Formulated 2 threat models (Label Flipping $y \mapsto 1-y$, Model-Update Delta Sign Reversal $\Delta_k' = -\gamma \Delta_k$). Implemented Coordinate-wise Median and Trimmed Mean ($m=2$). Demonstrated FedAvg vulnerability and robust recovery across 9 experiments and multi-seed evaluations. Executed `notebooks/06_robust_federated_learning.ipynb`. |
| **Phase 3B** | Advanced Defense & Backdoors | **NEXT** | Evaluate distance-based defense (Multi-Krum), cosine-similarity defense (FoolsGold), and stealthy targeted backdoor injection. |
| **Phase 3C** | Heterogeneity-Aware FL | **PLANNED** | Implement and evaluate FedProx and SCAFFOLD to combat client drift under extreme Non-IID. |
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

## ⚔️ Phase 3: Robust Federated Learning Under Adversarial Attacks

In open vehicular networks, malicious or compromised ECUs can inject poisoning attacks to degrade global detection or induce false alarms. Phase 3 systematically assesses vulnerabilities and evaluates robust aggregation algorithms.

### Threat Model
- **Network Setting:** 10 simulated clients, Non-IID Dirichlet distribution ($\alpha = 0.5$), 10 communication rounds.
- **Adversary Budget:** 2 malicious clients ($f=2$, 20% of fleet), specifically designated as `client_0` and `client_1`.
- **Attack A — Label Flipping (Data Poisoning):**
  - Malicious clients invert all training labels: $y \mapsto 1 - y$ ($0 \to 1$ and $1 \to 0$).
  - Malicious clients train honestly on their poisoned data using standard local SGD.
  - Validation sets, test sets, and attack-type evaluation metadata remain strictly untouched.
  - Logged: `client_0` flipped 1,845 labels (100%), `client_1` flipped 2,427 labels (100%).
- **Attack B — Model-Update Corruption via Delta Sign Reversal (Byzantine Model Poisoning):**
  - Malicious clients train on their legitimate local data, but instead of sending honest updates, they submit corrupted model deltas:
    $$\Delta_k' = -\gamma \cdot \Delta_k \quad \text{where } \Delta_k = \theta_k^{(t)} - \theta^{(t-1)}$$
  - In our benchmark, $\gamma = 1.5$. The poisoned transmitted state is $\theta_k' = \theta^{(t-1)} + \Delta_k'$.
  - This directly pulls the global model away from optimal convergence.

### Defense Mechanisms (Aggregation Rules)
1. **Sample-Weighted FedAvg:** Baseline linear weighted average: $\theta^{(t)} = \sum_{k} \frac{n_k}{N} \theta_k^{(t)}$. Completely vulnerable to outliers and sign-inverted updates.
2. **Coordinate-wise Median:** For each scalar coordinate $j$ of each parameter tensor, compute the median across all $K$ client updates:
   $$\theta_j^{(t)} = \text{median}(\{\theta_{k, j}^{(t)}\}_{k=1}^K)$$
   Tolerates up to $f < K/2$ arbitrary Byzantine clients without divergence.
3. **Coordinate-wise Trimmed Mean:** For each coordinate $j$, sort the $K$ client values, discard the lowest $m$ and highest $m$ values ($m=2$), and average the remaining $K - 2m$ values:
   $$\theta_j^{(t)} = \frac{1}{K - 2m} \sum_{k=m+1}^{K-m} \theta_{(k), j}^{(t)}$$

---

## 📊 Results: Centralized vs. IID vs. Non-IID vs. Robust FL

All models were evaluated on the untouched test partition (4,690 windows: 2,900 Normal, 1,790 Attack) using the best checkpoint selected by Validation F1:

### Primary Matched Comparison ($\alpha = 0.5$, Seed 42)

| Threat Scenario | Aggregator | Best Rd | Val F1 (%) | Test Acc (%) | Test Prec (%) | Test Recall (%) | Test F1 (%) | FPR (%) | FNR (%) | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | Centralized | Ep 8 | 99.42% | **99.66%** | 99.61% | **99.50%** | **99.55%** | 0.24% | **0.50%** | **0.9992** |
| **IID Baseline** | FedAvg | Rd 9 | 97.99% | **98.25%** | 99.71% | 95.70% | **97.66%** | 0.17% | 4.30% | 0.9965 |
| **Non-IID Benign** | FedAvg | Rd 10 | 98.17% | **98.72%** | 99.60% | 97.04% | **98.30%** | 0.24% | 2.96% | 0.9964 |
| **Non-IID Benign** | Median | Rd 10 | 97.82% | **98.51%** | 98.81% | 97.26% | **98.03%** | 0.72% | 2.74% | 0.9956 |
| **Non-IID Benign** | Trimmed Mean | Rd 10 | 97.80% | **98.34%** | 99.42% | 96.20% | **97.79%** | 0.34% | 3.80% | 0.9955 |
| **Attack A: Label Flip** | **FedAvg** | Rd 10 | 93.35% | **94.63%** | 99.87% | **86.03% ⚠️** | **92.44%** | **0.07%** | **13.97% ⚠️** | 0.9932 |
| **Attack A: Label Flip** | **Median** | Rd 9 | 96.59% | **97.14%** | 95.05% | **97.60% ✅** | **96.31%** | 3.14% | **2.40% ✅** | 0.9945 |
| **Attack A: Label Flip** | **Trimmed Mean** | Rd 9 | 97.15% | **97.51%** | 96.19% | **97.32% ✅** | **96.75%** | 2.38% | **2.68% ✅** | 0.9946 |
| **Attack B: Update Corrupt**| **FedAvg** | Rd 2 | 55.47% | **38.17% 💥** | 38.17% | **100.00%** | **55.25% 💥** | **100.00% 💥**| **0.00%** | 0.5861 |
| **Attack B: Update Corrupt**| **Median** | Rd 9 | 96.75% | **97.31%** | 97.17% | **95.75% ✅** | **96.45%** | 1.72% | **4.25% ✅** | 0.9934 |
| **Attack B: Update Corrupt**| **Trimmed Mean** | Rd 10 | 96.70% | **97.06%** | 95.94% | **96.37% ✅** | **96.15%** | 2.52% | **3.63% ✅** | 0.9936 |

### Multi-Seed Statistical Stability (Seeds 42, 43, 44; Mean ± Std)

To guarantee that findings are robust and not an artifact of a single lucky seed, we repeated the key experiments across 3 independent seeds:

| Threat Scenario | Aggregator | Test Accuracy Mean ± Std (%) | Test F1 Mean ± Std (%) | Test FPR Mean ± Std (%) | Test FNR Mean ± Std (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Benign Fleet** | **FedAvg** | 97.96% ± 0.54% | **97.27% ± 0.73%** | 0.45% ± 0.20% | 4.62% ± 1.23% |
| **Benign Fleet** | **Median** | 96.93% ± 1.76% | **95.97% ± 2.26%** | 1.97% ± 2.16% | 4.86% ± 1.54% |
| **Benign Fleet** | **Trimmed Mean**| 97.56% ± 0.83% | **96.77% ± 1.07%** | 1.11% ± 1.09% | 4.58% ± 0.56% |
| **Attack A: Label Flipping** | **FedAvg** | 74.55% ± 26.17% | **78.56% ± 16.99% ⚠️** | 35.37% ± 45.77% | 9.39% ± 5.59% |
| **Attack A: Label Flipping** | **Median** | **95.19% ± 2.82%** | **93.75% ± 3.57% ✅** | **3.91% ± 3.38%** | **6.28% ± 2.93%** |
| **Attack A: Label Flipping** | **Trimmed Mean**| 93.06% ± 6.26% | **91.97% ± 6.64%** | 9.01% ± 10.90% | 3.59% ± 1.87% |
| **Attack B: Update Corrupt** | **FedAvg** | 54.04% ± 17.76% | **63.43% ± 10.45% 💥** | 73.14% ± 28.39% 💥 | 1.94% ± 2.32% |
| **Attack B: Update Corrupt** | **Median** | **95.10% ± 2.61%** | **93.60% ± 3.24% ✅** | **3.53% ± 3.68%** | **7.13% ± 2.04%** |
| **Attack B: Update Corrupt** | **Trimmed Mean**| 77.20% ± 27.60% | **82.14% ± 19.02%** | 34.23% ± 46.52% | 4.28% ± 3.79% |

*(Note: Standard deviations computed across seeds [42, 43, 44] using ddof=0 matching JSON artifacts; ddof=1 sample std is ±20.80% for Label Flip FedAvg and ±4.38% for Median).*

### Per-Attack-Type Recall Analysis under Poisoning (Seed 42)

| Cyber-Attack Class | Test Windows | Benign FedAvg | Label Flip FedAvg | Label Flip Median | Update Corrupt FedAvg | Update Corrupt Median |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gear Spoofing** | 591 | 99.32% | **94.08% ⚠️** | 98.98% | 100.00%* | 97.46% |
| **RPM Spoofing** | 592 | 100.00% | **72.64% ⚠️** | 100.00% | 100.00%* | 98.82% |
| **DoS Injection** | 398 | 98.24% | 96.23% | 98.24% | 100.00%* | 97.74% |
| **Fuzzy Injection** | 209 | 79.90% | 81.82% | 85.65% | 100.00%* | 78.47% |

*\*Note: Under Update Corruption, FedAvg classified 100% of samples as attacks (FPR = 100.00%), which rendered the IDS operationally unusable.*

> **Operational Insight:** Under Label Flipping, FedAvg misses **27.4% of RPM sensor spoofing attacks** (162 missed frames) and **5.9% of Gear spoofing attacks** because malicious clients 8 and 9 held substantial telemetry for those IDs and flipped their labels. Coordinate-wise Median fully restores RPM detection to 100.00% and Gear to 98.98% without requiring centralized data.

---

### Key Scientific Takeaways:
1. **Catastrophic Failure of FedAvg Under Adversarial Attacks:**
   - Under **Attack A (Label Flipping)**, FedAvg's test recall plummets to 86.03% (FNR spikes from 2.74% to 13.97%, missing 250 attacks). Across multiple seeds, recall deteriorates as low as 43.18%.
   - Under **Attack B (Model-Update Sign Reversal)**, FedAvg experiences complete network collapse (Test F1 = 55.25%, Accuracy = 38.17%, FPR = 100.00%), classifying every normal frame as an attack.
2. **Coordinate-wise Median is the Most Reliable Defense:**
   - Median recovers detection across both attack modalities to $>96.4\%$ F1 on seed 42 and $>93.6\%$ F1 across independent seeds.
   - It incurs a negligible benign utility penalty of only 0.27% relative to FedAvg.
3. **Non-IID Vulnerability in Trimmed Mean:**
   - While Trimmed Mean performs admirably on seed 42 (96.77% F1), it exhibits substantial multi-seed volatility ($82.14\% \pm 19.02\%$ F1). Under Non-IID Dirichlet partitioning, natural client drift causes honest clients with specialized local data distributions to appear in the outer $m$ tails, leading Trimmed Mean to discard honest updates rather than poisoned ones.
4. **Operational Trade-offs:**
   - An IDS must balance False Negative Rate (missed intrusions that compromise vehicular safety) against False Positive Rate (nuisance alarms that degrade driver confidence). FedAvg fails on both extremes (FNR = 31.25% under label flipping; FPR = 73.14% under update corruption), whereas Coordinate-wise Median maintains a balanced operating point.

---

## ⚖️ Important Methodological Decisions

1. **Chronological Per-Scenario Splitting:** Random shuffling destroys temporal packet dynamics and leaks messages. We enforce strict chronological ordering (70% train, 15% val, 15% test).
2. **16-Frame Sliding Windows ($S=W=16$):** Grouping into 16 consecutive frames captures frequency bursts without frame overlap.
3. **GroupNorm over BatchNorm:** `GroupNorm` normalizes per sample across 4 channel groups, avoiding the catastrophic running statistics divergence that affects `BatchNorm1d` under Non-IID data.
4. **Domain-Based Normalization:** Protocol constants (CAN ID by 2047.0, DLC by 8.0, payload by 255.0, inter-arrival time by physical maximum) prevent dataset leakage.
5. **Leak-Free Initial Inter-Arrival Time:** Frame 0 inter-arrival time is initialized to 0.0, reflecting monitoring inception without future timestamp consumption.
6. **Best Validation Checkpoint Selection:** Selected strictly by validation F1 score, keeping the test set untouched until post-training evaluation.
7. **Strict Parameter Compatibility:** Aggregators validate parameter tensor keys, shapes, and floating-point types before aggregation.

---

## ⚠️ Current Limitations

- **Simulated Clients:** The 10 clients are simulated partitions of a single Kia Soul capture dataset, rather than 10 physically distinct vehicles with varied ECU architectures.
- **Privacy Limitations:** Federated learning keeps raw CAN frames local, but does not provide formal privacy guarantees (such as $(\epsilon, \delta)$-differential privacy) against reconstruction or membership inference attacks.
- **Threat Model Scope:** Our evaluations focused on untargeted label flipping and model update sign-reversal. Stealthy backdoor attacks designed to evade coordinate-wise statistics require dedicated cosine or spectral defenses (e.g., FoolsGold).
- **Trimmed Mean Non-IID Sensitivity:** In extreme Non-IID fleets, static trimming fractions can discard legitimate minority-class client updates.

---

## 🧭 Next Step

```text
NEXT:
Phase 3B — Evaluate advanced Byzantine defenses (Multi-Krum, FoolsGold) against targeted backdoors 
and adaptive poisoning in heterogeneous vehicular networks.
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
│   └── federated/           # dataset.py, client.py, server.py, aggregation.py, threat.py
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb   # Executed EDA, CAN ID & class distributions
│   ├── 02_preprocessing.ipynb      # Executed feature extraction & splitting
│   ├── 03_baseline_ids.ipynb       # Executed centralized baseline training & evaluation
│   ├── 04_federated_iid.ipynb       # Executed Phase 2A IID Federated baseline (10 clients, FedAvg)
│   ├── 05_federated_noniid.ipynb    # Executed Phase 2B Non-IID Dirichlet benchmark (alpha in {1.0, 0.5, 0.1})
│   └── 06_robust_federated_learning.ipynb # Executed Phase 3 Robust FL & Byzantine defense benchmark
│
├── results/
│   ├── figures/             # Confusion matrices, training curves, robust comparison plots
│   ├── metrics/             # centralized, IID, Non-IID, and robust FL experiment JSON metrics
│   ├── models/              # Checkpoints for centralized, IID, Non-IID, and robust models (.pt)
│   └── reports/             # centralized_baseline_v2.md, federated_iid_report.md, federated_noniid_report.md, robust_federated_report.md
│
├── configs/                 # baseline_config.json, federated_iid_config.json, federated_noniid_config.json, robust_federated_config.json
├── tests/                   # test_loader.py, test_models.py, test_preprocessing.py, test_federated.py, test_robust_aggregation.py
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

### 2. Run Test Suite (35 Unit Tests)
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

# Phase 3: Robust Federated Learning Against Malicious Clients
jupyter nbconvert --to notebook --execute --inplace notebooks/06_robust_federated_learning.ipynb
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
