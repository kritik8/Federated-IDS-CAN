# Phase 2A — Federated Learning IID Baseline Evaluation Report

**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Repository:** `Federated-IDS-CAN`  
**Milestone:** Phase 2A — 10-Client IID Federated Baseline with Federated Averaging (FedAvg)  
**Date:** October 2026  
**Status:** Certified Reproducible Federated Baseline  

---

## 1. Executive Summary

This report presents the experimental results of **Phase 2A**, establishing the decentralized Federated Learning (FL) baseline under an **Independent and Identically Distributed (IID)** traffic regime.

Using **10 simulated vehicular clients** and standard **Federated Averaging (FedAvg)** (McMahan et al., 2017), the global model achieved:
- **98.25% Test Accuracy** (4,608 / 4,690 correct)
- **97.66% Test F1-Score**
- **99.71% Test Precision**
- **95.70% Test Recall**
- **0.17% False Positive Rate (FPR)** (only 5 false alarms out of 2,900 normal windows)
- **4.30% False Negative Rate (FNR)** (77 missed attacks out of 1,790 attack windows)
- **0.9965 ROC-AUC**
- **Wall-Clock Training Time:** **35.45 seconds** (CPU, 10 rounds $\times$ 1 local epoch)
- **Communication Cost:** **158.63 KB** per model upload/download (**3.10 MB** total network exchange per round across all 10 clients)

This confirms that decentralized collaborative training without exchanging raw vehicular CAN packets can achieve near-centralized detection performance ($97.66\%$ vs $99.55\%$ centralized F1) under ideal IID conditions.

---

## 2. Experiment Setup & Hyperparameters

All hyperparameters adhere strictly to `configs/federated_iid_config.json`, maintaining exact alignment with the centralized baseline:

| Hyperparameter | Value | Description / Rationale |
|---|---|---|
| **Architecture** | `CAN1DCNN` | 3 Conv1D layers (32, 64, 128), GroupNorm ($G=4$), FC(64), Dropout(0.2) |
| **Model Parameters** | 40,610 | 100% trainable float32 weights & biases; 0 running buffers |
| **Simulated Clients ($K$)** | 10 | Simulated vehicular electronic control units (ECUs) / connected vehicles |
| **Client Participation** | 100% (10/10) | Full client participation per communication round |
| **Data Partitioning** | IID Stratified | Exact class-ratio preservation; 0 sample overlap, 0 omissions |
| **Aggregation Algorithm** | Sample-Weighted FedAvg | $\theta_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N} \theta_k$ |
| **Communication Rounds ($T$)** | 10 | Global coordination cycles |
| **Local Epochs per Round ($E$)** | 1 | Single pass over local client dataset per round |
| **Client Optimizer** | Adam | Learning rate $\eta = 0.001$, Weight decay $\lambda = 10^{-4}$ |
| **Local Batch Size** | 128 | Consistent with centralized baseline mini-batch size |
| **Loss Function** | CrossEntropyLoss | Standard multi-class / binary cross-entropy loss |
| **Random Seed** | 42 | Set for PyTorch, NumPy, and random partitioners |
| **Execution Platform** | CPU | Windows 11, Intel Core, PyTorch 2.9.1+cpu |

---

## 3. Client Dataset Partitioning & Class Distributions

The centralized training pool (21,875 sliding windows of size $16 \times 11$) was partitioned deterministically using `partition_iid_stratified(seed=42)` across 10 simulated clients.

### Verification Checklist:
- **Completeness:** $\sum_{k=0}^9 n_k = 21,875$ (100% accounted for).
- **Mutual Exclusivity:** Zero index overlap between any pair of clients ($\bigcap_{k} S_k = \emptyset$).
- **Balance:** 7 clients hold 2,187 samples, 1 client holds 2,188 samples, 2 clients hold 2,189 samples (maximum size difference: 2 samples).
- **Attack Proportions:** Each client possesses between $39.09\%$ and $39.12\%$ attack traffic (matching the training pool ground truth of $39.10\%$).

### Client Traffic Breakdown:

| Client ID | Total Samples ($n_k$) | Normal ($y=0$) | Attack ($y=1$) | Attack Ratio | DoS | Fuzzy | Gear | RPM |
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
| **Total** | **21,875** | **13,322** | **8,553** | **39.10%** | **1,920** | **1,590** | **2,508** | **2,535** |

*Note on Client Scope:* In this simulation, each client models a simulated vehicle / ECU within a coordinated fleet. Validation (4,685 windows) and Test (4,690 windows) sets remain strictly global and are never split among clients.

---

## 4. Communication Overhead Analysis

Communication cost is calculated analytically based on serialized PyTorch `state_dict` tensor storage (IEEE 754 32-bit floating-point numbers). It does not represent physical network interface socket measurements:

- **Model Parameter Count:** 40,610 parameters
- **Raw Parameter Size:** $40,610 \times 4 \text{ bytes} = 162,440\text{ bytes} \approx 158.63\text{ KB}$
- **Downlink per Round (Server $\to$ 10 Clients):** $10 \times 158.63\text{ KB} = 1,586.33\text{ KB}$ ($\approx 1.55\text{ MB}$)
- **Uplink per Round (10 Clients $\to$ Server):** $10 \times 158.63\text{ KB} = 1,586.33\text{ KB}$ ($\approx 1.55\text{ MB}$)
- **Total Round Communication:** **3,172.66 KB** ($\approx 3.10\text{ MB}$)
- **10-Round Total Cumulative Traffic:** **30.98 MB**

### Automotive Feasibility:
In a real-world connected vehicular environment, a client payload of $158.6\text{ KB}$ per coordination round can easily be transmitted over standard Cellular V2X (C-V2X), LTE-M, or 5G telemetry links without saturating the telematics control unit (TCU).

---

## 5. Round-by-Round Training Dynamics

Tracking global model convergence on the held-out validation set (4,685 windows) after each aggregation round:

| Round | Mean Client Train Loss | Global Val Loss | Global Val Acc (%) | Global Val F1 (%) | Checkpoint Selection |
|---|---|---|---|---|---|
| **Round 1** | 0.6051 | 0.3946 | 89.58% | 84.42% | Initial baseline |
| **Round 2** | 0.3487 | 0.1597 | 95.92% | 94.63% | +10.21% F1 |
| **Round 3** | 0.2123 | 0.0998 | 97.25% | 96.30% | +1.67% F1 |
| **Round 4** | 0.2875 | 0.1278 | 95.52% | 93.80% | Transient perturbation |
| **Round 5** | 0.1971 | 0.0948 | 97.01% | 95.95% | Recovery |
| **Round 6** | 0.1989 | 0.0941 | 96.99% | 95.92% | Plateau |
| **Round 7** | 0.1540 | 0.0689 | 98.06% | 97.41% | +1.49% F1 |
| **Round 8** | 0.1401 | 0.0623 | 98.21% | 97.61% | +0.20% F1 |
| **Round 9** | **0.1133** | **0.0548** | **98.48%** | **97.99%** | ⭐ **BEST CHECKPOINT (Restored)** |
| **Round 10** | 0.1100 | 0.0578 | 98.34% | 97.79% | Slight validation dip |

**Best Checkpoint Decision:** Following the strict protocol, checkpoint selection is driven exclusively by **Validation F1-score**. Round 9 achieved the highest Validation F1 ($97.9904\%$), and its model parameters were restored for final test set evaluation.

---

## 6. Evaluation on the Untouched Test Set

The restored Round 9 global model was evaluated **once** on the untouched chronological test set (4,690 windows: 2,900 Normal, 1,790 Attack):

### Comprehensive Test Metrics:

| Metric | Measured Value | Counts / Details |
|---|---|---|
| **Accuracy** | **98.25%** (0.982516) | 4,608 correct out of 4,690 |
| **Precision** | **99.71%** (0.997090) | $TP / (TP + FP) = 1,713 / (1,713 + 5)$ |
| **Recall (Sensitivity)** | **95.70%** (0.956983) | $TP / (TP + FN) = 1,713 / (1,713 + 77)$ |
| **F1-Score** | **97.66%** (0.976625) | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **0.17%** (0.001724) | 5 false alarms out of 2,900 Normal windows |
| **False Negative Rate (FNR)** | **4.30%** (0.043017) | 77 missed attacks out of 1,790 Attack windows |
| **ROC-AUC** | **0.9965** (0.996487) | Area under ROC curve |
| **Test Loss** | **0.0583** | CrossEntropyLoss on test set |

### Confusion Matrix:

```
                   Predicted Normal    Predicted Attack       Total
Actual Normal            2,895                 5              2,900
Actual Attack               77             1,713              1,790
Total                    2,972             1,718              4,690
```

### Analysis of Missed Attacks by Cyber-Attack Class:

| Traffic Class | Total Test Windows | Correct Detections | Missed (Errors) | Class Detection Rate (%) |
|---|---|---|---|---|
| **Normal Traffic** | 2,900 | 2,895 | 5 (FP) | **99.83%** |
| **DoS Injection** | 398 | 390 | 8 (FN) | **97.99%** |
| **Gear Spoofing** | 591 | 577 | 14 (FN) | **97.63%** |
| **RPM Spoofing** | 592 | 577 | 15 (FN) | **97.47%** |
| **Fuzzy Injection** | 209 | 169 | 40 (FN) | **80.86%** |
| **Overall** | **4,690** | **4,608** | **82** | **98.25%** |

**Insight into Fuzzy Attack Detection:**  
The Fuzzy attack injects random CAN IDs with randomized payloads at irregular intervals. In a decentralized training configuration where each client sees only 1/10th of the training data (approx. 150–160 Fuzzy windows per client) and local training is restricted to 1 local epoch per round, the global model requires additional rounds or higher local capacity to fully learn the sparse boundary between arbitrary normal telemetry and randomized malicious IDs. Nevertheless, high-volume DoS and semantic Spoofing attacks exceed $97.5\%$ recall.

---

## 7. Direct Comparison: Centralized vs. Federated Baseline

| Metric / Dimension | Centralized Baseline (`v2`) | Federated IID FedAvg | Performance Delta | Analysis & Commentary |
|---|---|---|---|---|
| **Test Accuracy** | **99.66%** | **98.25%** | **-1.41%** | Negligible accuracy loss without data pooling |
| **Test Precision** | 99.61% | **99.71%** | **+0.10%** | Federated model produces fewer false alarms (5 vs 7) |
| **Test Recall** | **99.50%** | 95.70% | -3.80% | 77 missed attack windows vs 9 in centralized |
| **Test F1-Score** | **99.55%** | **97.66%** | **-1.89%** | Excellent retention of detection capability |
| **False Positive Rate (FPR)** | 0.24% | **0.17%** | **-0.07%** | Ultra-low false alarm rate (vital for automotive ECUs) |
| **False Negative Rate (FNR)** | **0.50%** | 4.30% | +3.80% | Concentrated primarily in subtle Fuzzy injections |
| **ROC-AUC** | **0.9992** | 0.9965 | -0.0027 | Both curves demonstrate exceptional class separation |
| **Data Privacy** | ❌ Raw CAN logs uploaded | ✅ Zero raw data shared | **Privacy Preserved** | Eliminates privacy and telemetry leakage |
| **Network Bandwidth** | ❌ Huge raw CSV uploads | ✅ 158.6 KB model updates | **Drastic Reduction** | Efficient edge vehicular deployment |
| **Training Budget** | 10 central epochs | 10 rounds $\times$ 1 epoch | Equivalent Volume | Same sample volume (21,875 samples/epoch) |
| **Training Duration** | 25.12 seconds | 35.45 seconds | +10.33 seconds | Additional overhead from parameter broadcast & FedAvg |

### Algorithmic Differences in Optimization:
In the centralized setting, a single optimizer executes continuous mini-batch SGD over all 21,875 samples in shuffled order. In the federated setting, 10 distinct local optimizers step on their respective local subsets for 1 epoch before parameter coordinates are averaged. Despite this decentralized gradient fragmentation, FedAvg achieves $97.66\%$ F1 under IID conditions.

---

## 8. Verification & Test Suite Results

The federated implementation was rigorously tested with unit and integration tests.

### Test Coverage (`tests/test_federated.py`):
1. `test_iid_partition_completeness`: Confirms all 21,875 indices are assigned.
2. `test_iid_partition_no_overlap`: Confirms zero duplicate or overlapping indices.
3. `test_iid_partition_reproducibility`: Confirms deterministic partitioning under identical seeds.
4. `test_iid_partition_class_balance`: Asserts attack ratio variance across clients is $<1\%$.
5. `test_fedavg_synthetic_weights`: Verifies numerical sample-weighting on known synthetic tensors.
6. `test_fedavg_state_dict_validation`: Validates rejection of mismatched shapes and keys.
7. `test_identical_initialization`: Verifies all clients receive bit-identical broadcast parameters.
8. `test_end_to_end_one_round_training`: Executes a synthetic 1-round train/aggregation cycle.
9. `test_communication_cost_calculation`: Verifies analytic byte and KB communication math.

### Full Test Suite Execution:
```bash
python -m unittest discover tests
..................
----------------------------------------------------------------------
Ran 18 tests in 3.680s

OK
```
All **18 tests** (9 centralized + 9 federated) passed without warnings or errors.

---

## 9. Limitations & Next Research Phase

### Current Limitations of Phase 2A:
1. **Idealized Traffic (IID Assumption):** Real vehicular fleets do not produce identically distributed traffic. Different ECUs (e.g., Engine Gateway, Infotainment, ADAS) and different vehicle models exhibit severe statistical heterogeneity.
2. **Honest Clients:** All 10 clients are assumed fully cooperative and benign. No poisoning or Byzantine noise was present.
3. **Synchronous Scheduling:** All 10 clients participate simultaneously without stragglers or packet dropouts.

### Next Milestone — Phase 2B (Controlled Non-IID Evaluation):
In Phase 2B, we will systematically break the IID assumption by implementing Dirichlet distribution partitioning ($\text{Dir}(\alpha)$ for $\alpha \in \{0.1, 0.5, 1.0\}$) across the 10 clients, quantifying the degree of **client drift** and performance degradation in standard FedAvg before evaluating robust aggregators.
