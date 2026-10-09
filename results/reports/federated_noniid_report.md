# Phase 2B — Non-IID Federated Learning Experiments Evaluation Report

**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Repository:** `Federated-IDS-CAN`  
**Milestone:** Phase 2B — Controlled Non-IID Dirichlet Partitioning ($\alpha \in \{1.0, 0.5, 0.1\}$) under FedAvg  
**Date:** October 2026  
**Status:** Certified Reproducible Non-IID Benchmark  

---

## 1. Executive Summary

This report presents the experimental results of **Phase 2B**, evaluating the impact of **fleet data heterogeneity (Non-IID traffic)** on decentralized collaborative intrusion detection under standard **Federated Averaging (FedAvg)**.

Using an attack-type-aware **symmetric Dirichlet allocation** ($\text{Dir}(\alpha \cdot \mathbf{1}_K)$) across 10 simulated vehicular clients while retaining binary intrusion detection targets ($y \in \{0, 1\}$), we systematically evaluated three distinct heterogeneity regimes alongside the established IID control baseline:
1. **Moderate Heterogeneity ($\alpha = 1.0$):** **98.51% Accuracy**, **98.01% F1-score**, **0.10% FPR**, **3.74% FNR**, **0.9961 ROC-AUC**.
2. **High Heterogeneity ($\alpha = 0.5$):** **98.49% Accuracy**, **97.98% F1-score**, **0.24% FPR**, **3.58% FNR**, **0.9956 ROC-AUC**.
3. **Extreme Heterogeneity ($\alpha = 0.1$):** **92.11% Accuracy**, **88.69% F1-score**, **1.07% FPR**, **18.94% FNR**, **0.9573 ROC-AUC**.

### Key Empirical Findings:
- **Resilience to Moderate Skew:** Under moderate and high heterogeneity ($\alpha = 1.0$ and $\alpha = 0.5$), FedAvg with `GroupNorm` maintains remarkable stability, matching or slightly exceeding the IID baseline ($97.66\%$ F1) on the global test set.
- **Catastrophic Client Drift under Extreme Skew ($\alpha = 0.1$):** When local clients experience severe label and category isolation (several clients observing 100% attack traffic and 0 normal frames, while others observe >90% normal frames), standard FedAvg suffers from severe gradient conflict:
  - Global F1 drops by **-8.97%** ($97.66\% \to 88.69\%$).
  - Detection Recall collapses from **95.70%** to **81.06%** (a **14.64% drop**).
  - False Negative Rate surges from **4.30%** to **18.94%** (more than a **4x increase** in missed cyber-attacks).
  - High-volume **DoS detection collapses from 97.99% to 43.97%**, and subtle **Fuzzy attack detection collapses from 80.86% to 44.50%**.

---

## 2. Partitioning Methodology & Mathematical Formulation

### 2.1 Attack-Type-Aware Dirichlet Allocation
In vehicular CAN networks, an intrusion detection system operates as a binary classifier (Normal vs. Attack). However, cyber-threats consist of distinct semantic and temporal attack vectors (DoS, Fuzzy, Gear spoofing, RPM spoofing). In real vehicular fleets, different vehicles experience different threat profiles.

To model this realistic phenomenon, we partition the 21,875 training windows using the underlying category metadata:
- $\mathcal{C} = \{\text{Normal}, \text{DoS}, \text{Fuzzy}, \text{Gear}, \text{RPM}\}$
- For each category $c \in \mathcal{C}$ with $N_c$ total samples:
  1. A multinomial allocation probability vector is drawn:
     $$\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K), \quad \sum_{k=0}^{K-1} p_{c, k} = 1.0$$
  2. Proportions are converted into integer sample counts:
     $$n_{c, k} = \lfloor p_{c, k} \times N_c \rfloor$$
     with fractional residuals $(p_{c, k} \times N_c) - n_{c, k}$ allocated to the largest remaining clients to guarantee $\sum_{k=0}^{K-1} n_{c, k} = N_c$ exactly.
  3. Deterministically shuffled category indices are assigned to client $k$.
  4. Each assigned sample retains its binary target:
     $$y = \begin{cases} 0, & \text{if category is Normal} \\ 1, & \text{if category is DoS, Fuzzy, Gear, or RPM} \end{cases}$$

### 2.2 Rejection Sampling & Feasibility Constraints
In continuous Dirichlet draws with very small concentration parameters ($\alpha = 0.1$), probability mass can concentrate on fewer than $K$ clients, leaving one or more simulated clients with 0 samples. An empty client cannot perform local mini-batch gradient descent.

To ensure all $K=10$ clients participate in local training, the partitioner enforces `min_samples_per_client = 64` (half of `batch_size = 128`) via deterministic rejection sampling. With `seed = 42`:
- $\alpha = 1.0$: Trial 0 satisfies the constraint ($\min(n_k) = 729$).
- $\alpha = 0.5$: Trial 0 satisfies the constraint ($\min(n_k) = 1,004$).
- $\alpha = 0.1$: Trial 1 satisfies the constraint ($\min(n_k) = 196$).

### 2.3 Methodological Guarantees:
1. **Training Pool Isolation:** Partitioning operates solely on the 21,875 training samples.
2. **Untouched Validation & Test Sets:** Global validation (4,685 windows) and test sets (4,690 windows) remain undivided on the server.
3. **Completeness & Exclusivity:** Exactly 21,875 samples assigned; zero index duplicates; zero index omissions.
4. **Deterministic Reproducibility:** Governed strictly by `seed = 42`.

---

## 3. Client Sample Distributions across Regimes

### 3.1 Non-IID $\alpha = 1.0$ (Moderate Skew)

| Client | Total Samples | Normal ($y=0$) | Attack ($y=1$) | Attack % | DoS | Fuzzy | Gear | RPM |
|---|---|---|---|---|---|---|---|---|
| **Client 0** | 1,296 | 581 | 715 | 55.17% | 118 | 138 | 136 | 323 |
| **Client 1** | 2,882 | 1,361 | 1,521 | 52.78% | 144 | 17 | 109 | 1,251 |
| **Client 2** | 2,375 | 2,160 | 215 | 9.05% | 5 | 41 | 73 | 96 |
| **Client 3** | 729 | 428 | 301 | 41.29% | 82 | 52 | 162 | 5 |
| **Client 4** | 1,283 | 790 | 493 | 38.43% | 152 | 190 | 111 | 40 |
| **Client 5** | 1,034 | 605 | 429 | 41.49% | 148 | 38 | 110 | 133 |
| **Client 6** | 4,286 | 3,480 | 806 | 18.81% | 256 | 28 | 504 | 18 |
| **Client 7** | 3,212 | 1,548 | 1,664 | 51.81% | 173 | 369 | 755 | 367 |
| **Client 8** | 3,391 | 2,114 | 1,277 | 37.66% | 788 | 1 | 290 | 198 |
| **Client 9** | 1,387 | 255 | 1,132 | 81.62% | 54 | 716 | 258 | 104 |
| **Total** | **21,875** | **13,322** | **8,553** | **39.10%** | **1,920** | **1,590** | **2,508** | **2,535** |

### 3.2 Non-IID $\alpha = 0.5$ (High Skew)

| Client | Total Samples | Normal ($y=0$) | Attack ($y=1$) | Attack % | DoS | Fuzzy | Gear | RPM |
|---|---|---|---|---|---|---|---|---|
| **Client 0** | 1,296 | 445 | 851 | 65.66% | 182 | 0 | 181 | 488 |
| **Client 1** | 1,347 | 679 | 668 | 49.59% | 1 | 41 | 624 | 2 |
| **Client 2** | 2,731 | 2,049 | 682 | 24.97% | 264 | 2 | 210 | 206 |
| **Client 3** | 2,629 | 951 | 1,678 | 63.83% | 609 | 823 | 140 | 106 |
| **Client 4** | 1,689 | 1,152 | 537 | 31.79% | 89 | 336 | 12 | 100 |
| **Client 5** | 1,078 | 241 | 837 | 77.64% | 149 | 199 | 464 | 25 |
| **Client 6** | 1,004 | 707 | 297 | 29.58% | 38 | 56 | 124 | 79 |
| **Client 7** | 1,428 | 1,044 | 384 | 26.89% | 113 | 27 | 70 | 174 |
| **Client 8** | 3,572 | 2,536 | 1,036 | 29.00% | 294 | 0 | 40 | 702 |
| **Client 9** | 5,101 | 3,518 | 1,583 | 31.03% | 181 | 106 | 643 | 653 |
| **Total** | **21,875** | **13,322** | **8,553** | **39.10%** | **1,920** | **1,590** | **2,508** | **2,535** |

### 3.3 Non-IID $\alpha = 0.1$ (Extreme Skew)

| Client | Total Samples | Normal ($y=0$) | Attack ($y=1$) | Attack % | DoS | Fuzzy | Gear | RPM |
|---|---|---|---|---|---|---|---|---|
| **Client 0** | 611 | 90 | 521 | 85.27% | 0 | 81 | 440 | 0 |
| **Client 1** | 196 | 0 | 196 | **100.00%** | 2 | 4 | 0 | 190 |
| **Client 2** | 1,700 | 121 | 1,579 | 92.88% | 161 | 1,033 | 0 | 385 |
| **Client 3** | 9,660 | 9,048 | 612 | 6.34% | 0 | 0 | 329 | 283 |
| **Client 4** | 461 | 0 | 461 | **100.00%** | 283 | 178 | 0 | 0 |
| **Client 5** | 1,982 | 595 | 1,387 | 69.98% | 1,241 | 146 | 0 | 0 |
| **Client 6** | 1,618 | 80 | 1,538 | 95.06% | 145 | 0 | 1,378 | 15 |
| **Client 7** | 1,571 | 0 | 1,571 | **100.00%** | 55 | 0 | 0 | 1,516 |
| **Client 8** | 2,360 | 2,186 | 174 | 7.37% | 14 | 146 | 0 | 14 |
| **Client 9** | 1,716 | 1,202 | 514 | 29.95% | 19 | 2 | 361 | 132 |
| **Total** | **21,875** | **13,322** | **8,553** | **39.10%** | **1,920** | **1,590** | **2,508** | **2,535** |

*Notable Skew Properties in $\alpha=0.1$:*
- **Complete Label Skew:** Clients 1, 4, and 7 possess **0 Normal samples** (100% attack traffic).
- **Dominant Normal Client:** Client 3 holds **9,048 Normal samples** (67.9% of all training normal traffic) and zero DoS or Fuzzy attacks.

---

## 4. FedAvg Training Dynamics across Rounds

Global model convergence on the held-out validation set (4,685 windows):

| Round | IID Baseline Val F1 | Non-IID $\alpha=1.0$ Val F1 | Non-IID $\alpha=0.5$ Val F1 | Non-IID $\alpha=0.1$ Val F1 |
|---|---|---|---|---|
| **Round 1** | 84.42% | 85.04% | 87.28% | 70.04% |
| **Round 2** | 94.63% | 93.68% | 92.64% | **54.18% (Loss: 1.57)** |
| **Round 3** | 96.30% | 96.08% | 95.25% | 81.17% |
| **Round 4** | 93.80% | 96.06% | 96.34% | 66.58% |
| **Round 5** | 95.95% | 96.90% | 96.70% | 83.76% |
| **Round 6** | 95.92% | 97.41% | 97.36% | 80.04% |
| **Round 7** | 97.41% | 97.68% | 97.56% | 87.25% |
| **Round 8** | 97.61% | 97.74% | 97.59% | 87.79% |
| **Round 9** | **97.99%** (Best) | 97.87% | **97.94%** (Best) | 89.13% |
| **Round 10** | 97.79% | **98.05%** (Best) | 97.82% | **90.12%** (Best) |

### Optimization Dynamics Commentary:
- In $\alpha=1.0$ and $\alpha=0.5$, convergence closely mimics the IID baseline, climbing monotonically into the $97.8\% - 98.0\%$ F1 band.
- In $\alpha=0.1$, Round 2 suffered a **catastrophic divergence**: validation loss spiked to **1.5674** and validation F1 plunged to **54.18%** (validation accuracy dropped to 39.42%). This occurred because the 3 pure-attack clients pushed network output heads aggressively toward attack logits, while Client 3 pushed heavily toward normal logits. Parameter averaging in FedAvg caused extreme interference. The model subsequently recovered to 90.12% F1 by Round 10, but permanently lost fine-grained decision boundaries for subtle attacks.

---

## 5. Evaluation on Untouched Test Set (4,690 Windows)

Each regime's best global checkpoint (selected strictly by Validation F1) was evaluated on the untouched test partition:

| Metric | Centralized (Ref) | IID Baseline | Non-IID $\alpha=1.0$ | Non-IID $\alpha=0.5$ | Non-IID $\alpha=0.1$ |
|---|---|---|---|---|---|
| **Best Round** | Epoch 8 | Round 9 | Round 10 | Round 9 | Round 10 |
| **Best Val F1** | 99.42% | 97.99% | 98.05% | 97.94% | **90.12%** |
| **Test Accuracy** | **99.66%** | **98.25%** | **98.51%** | **98.49%** | **92.11% (-6.14%)** |
| **Test Precision** | 99.61% | 99.71% | **99.83%** | 99.60% | 97.91% |
| **Test Recall** | **99.50%** | **95.70%** | **96.26%** | **96.42%** | **81.06% (-14.64%)** |
| **Test F1-Score** | **99.55%** | **97.66%** | **98.01%** | **97.98%** | **88.69% (-8.97%)** |
| **FPR** | 0.24% | 0.17% | **0.10%** | 0.24% | **1.07% (+0.90%)** |
| **FNR** | **0.50%** | 4.30% | 3.74% | 3.58% | **18.94% (+14.64%)** |
| **ROC-AUC** | **0.9992** | 0.9965 | 0.9961 | 0.9956 | **0.9573 (-0.0392)** |
| **True Negatives** | 2,893 | 2,895 | 2,897 | 2,893 | 2,869 |
| **False Positives** | 7 | 5 | **3** | 7 | 31 |
| **False Negatives** | 9 | 77 | 67 | 64 | **339** |
| **True Positives** | 1,781 | 1,713 | 1,723 | 1,726 | 1,451 |
| **Wall-Clock Time** | 25.12s | 35.45s | 28.73s | 25.51s | 27.41s |

---

## 6. Per-Attack-Type Intrusion Detection Breakdown

To pinpoint the exact cyber-security failure modes caused by client drift, we broke down test set accuracy across the individual traffic classes:

| Traffic Class | Total Test Windows | IID Detection | Non-IID $\alpha=1.0$ | Non-IID $\alpha=0.5$ | Non-IID $\alpha=0.1$ |
|---|---|---|---|---|---|
| **Normal Traffic** | 2,900 | 99.83% (2,895/2,900) | 99.90% (2,897/2,900) | 99.76% (2,893/2,900) | **98.93%** (2,869/2,900) |
| **Gear Spoofing** | 591 | 97.63% (577/591) | 99.32% (587/591) | 98.48% (582/591) | **100.00%** (591/591) |
| **RPM Spoofing** | 592 | 97.47% (577/592) | 99.32% (588/592) | 100.00% (592/592) | **100.00%** (592/592) |
| **DoS Injection** | 398 | 97.99% (390/398) | 97.99% (390/398) | 98.24% (391/398) | **43.97% (175/398) ⚠️** |
| **Fuzzy Injection** | 209 | 80.86% (169/209) | 75.60% (158/209) | 77.03% (161/209) | **44.50% (93/209) ⚠️** |

### Crucial Security Insights:
1. **Spoofing Immunity:** Gear and RPM spoofing attacks are detected with **100% accuracy** even under extreme Non-IID conditions ($\alpha = 0.1$). This is because sensor spoofing modifies fixed payload offsets and CAN arbitration IDs (`0x0316`, `0x043F`) that preserve recognizable structural patterns regardless of client label skew.
2. **The DoS Collapse:** In $\alpha = 0.1$, Client 5 held 1,241 DoS samples (64.6% of all training DoS samples), while 6 other clients had 0 or negligible DoS samples. When Client 5's weights were averaged into the global model, the high-frequency arbitration signature of DoS was diluted by normal-traffic gradients from Client 3 (9,048 samples). Consequently, the global model missed **223 out of 398 DoS attack windows** (56% miss rate).
3. **The Fuzzy Attack Vulnerability:** Fuzzy attacks inject random arbitration IDs and payloads. In $\alpha=0.1$, the global model failed to distinguish random ID injections from ambient normal traffic, missing **116 out of 209 Fuzzy attack windows** (55.5% miss rate).

---

## 7. Communication Overhead Comparison

Because all experiments used the identical `CAN1DCNN` architecture (40,610 trainable float32 parameters), the communication payload per round remains identical across all regimes:

- **Model Parameter Count:** 40,610 parameters
- **Per-Client State Dict Size:** **158.63 KB** (162,440 bytes)
- **Downlink per Round (10 Clients):** **1.55 MB** (1,586.33 KB)
- **Uplink per Round (10 Clients):** **1.55 MB** (1,586.33 KB)
- **Total Network Traffic per Round:** **3.10 MB** (3,172.66 KB)
- **Total 10-Round Network Exchange:** **30.98 MB**

*Note on Privacy & Communication:* All communication figures are derived from analytical serialized tensor state sizes. The simulation keeps raw training samples local to each client; however, vanilla parameter sharing without differential privacy does not formally guarantee zero information leakage against gradient inversion attacks.

---

## 8. Verification & Test Suite Status

The test suite in [`tests/test_federated.py`](file:///c:/Users/Kratik/IIIT%20Academics/Hustle/Major_Project_CAN_IDS/tests/test_federated.py) was expanded with `TestFederatedNonIID` to validate Dirichlet partitioning:
- `test_dirichlet_partition_completeness_and_no_overlap`: Verified for $\alpha \in \{1.0, 0.5, 0.1\}$.
- `test_dirichlet_deterministic_reproducibility`: Verified with seed matching.
- `test_dirichlet_alpha_validation_and_errors`: Verified rejection of $\alpha \le 0$ and impossible sample constraints.
- `test_dirichlet_client_distribution_summaries`: Verified dictionary shapes and sample totals.
- `test_dirichlet_binary_target_preservation`: Verified local targets strictly in $\{0, 1\}$.
- `test_dirichlet_attack_type_alignment`: Verified attack types map to label 1 and Normal maps to label 0.
- `test_dirichlet_small_categories_handling`: Verified edge cases on small class distributions.

### Test Execution:
```bash
python -m unittest discover tests
.........................
Ran 25 tests in 4.084s
OK
```
All **25 unit tests** (9 centralized + 16 federated) passed.

---

## 9. Limitations & Transition to Phase 3

### Current Limitations:
1. **Benign Clients Only:** All 10 clients are cooperative and honest. No malicious adversarial noise, label-flipping, or model poisoning has been introduced yet.
2. **Algorithm Drift:** FedAvg has no proximal regularization term (FedProx) or client control variates (SCAFFOLD) to correct client drift under $\alpha=0.1$.
3. **Synchronous Execution:** All clients participate in each round without dropouts or stragglers.

### Next Milestone — Phase 3 (Byzantine Threat Model & Robust Aggregation):
Now that Phase 2B has quantified the performance degradation induced by natural fleet heterogeneity, Phase 3 will introduce:
1. **Formal Adversarial Threat Models:** Malicious clients executing label-flipping, additive Gaussian noise, and targeted backdoor attacks.
2. **Byzantine-Robust Aggregation:** Replacing FedAvg with robust aggregation rules (Coordinate-wise Median, Trimmed Mean, Krum, FoolsGold) to secure vehicular intrusion detection against compromised connected vehicles.
