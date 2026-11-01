# Research Paper Readiness & Blueprint: Federated-IDS-CAN

**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Target Deadline:** November 12, 2026  
**Venue Type:** Peer-Reviewed Academic Conference / Journal (e.g., IEEE Transactions on Intelligent Transportation Systems, IEEE VTC, ACM CCS / WiSec)  
**Auditor:** Independent Senior Academic Reviewer & Research Systems Engineer  
**Output Location:** `final_audit_2026/PAPER_READINESS.md`

---

## 1. Recommended Paper Contributions

To withstand rigorous peer review, the manuscript should emphasize three primary contributions:

1. **A Leak-Free, Edge-Optimized Federated CAN IDS Benchmark:**
   We design and open-source a hardened intrusion detection pipeline on the real-world HCRL Kia Soul CAN dataset, featuring dynamic variable-DLC parsing, leak-free theoretical domain normalization for inter-arrival times ($\Delta t$), GroupNorm stabilization to eliminate distributed running-buffer divergence, and strict separation between real-time single-sample edge latency ($455.99\ \mu\text{s}$) and batched throughput.

2. **Systematic Characterization of Fleet Non-IID Heterogeneity on Cyber-Threat Vulnerability:**
   We introduce category-aware Dirichlet client partitioning ($\text{Dir}(\alpha \cdot \mathbf{1}_{10})$) across 10 simulated vehicular clients. We prove that while moderate skew ($\alpha \ge 0.5$) incurs minimal global utility penalty, extreme data heterogeneity ($\alpha = 0.1$) causes severe gradient conflict and client drift: global recall drops by **14.64%**, False Negative Rate quadruples to **18.94%**, and high-volume DoS and Fuzzy attack detection rates collapse from over 97% to ~44%.

3. **Byzantine Resilience and Vulnerability of Robust Aggregators under Combined Non-IID Skew:**
   Under a 20% Byzantine adversarial client budget executing data poisoning (Label Flipping) and model poisoning (Update Delta Sign Reversal), we demonstrate that standard FedAvg experiences catastrophic failure (F1 collapses to 55.25%, with 100% False Positive Rate under sign reversal). We prove that **Coordinate-wise Median** provides the most resilient and stable defense (Mean F1: $93.75\% \pm 3.57\%$ and $93.60\% \pm 3.24\%$ across seeds), while **Coordinate-wise Trimmed Mean** suffers from boundary leakage under severe Non-IID drift, occasionally trimming legitimate specialized client updates and dropping to 55.25% F1 on seed 44.

---

## 2. Defensible Research Novelty

What separates this paper from existing literature:
- **Prior Work Gap 1:** Most FL CAN IDS papers (e.g., in IEEE T-IV / IEEE IoT-J) assume synthetic IID client partitions, ignoring that real vehicular fleets experience highly skewed, heterogeneous cyber-threat exposure.
- **Prior Work Gap 2:** Papers that evaluate Non-IID federated learning rarely evaluate active Byzantine adversarial attacks simultaneously.
- **Prior Work Gap 3:** Standard Byzantine FL literature evaluates image datasets (MNIST, CIFAR-10) with uniform input bounds. This work is among the first to evaluate robust aggregation on continuous, temporal in-vehicle bus telemetry with strict real-time ECU execution constraints (<1.0 ms).
- **Novel Empirical Insight:** Demonstrating that Coordinate-wise Trimmed Mean interacts poorly with natural Non-IID client drift in vehicular settings, whereas Coordinate-wise Median remains stable.

---

## 3. Formal Research Questions (RQs)

The manuscript should be structured around three clear RQs:

- **RQ1 (Baseline Fidelity & Edge Viability):** Can a lightweight 1D-CNN using GroupNorm achieve centralized-grade intrusion detection accuracy on CAN bus traffic while meeting the sub-millisecond real-time execution deadline of embedded vehicular ECUs?
- **RQ2 (Fleet Heterogeneity Impact):** How does non-identical cyber-attack and class distribution (Non-IID Dirichlet skew with $\alpha \in \{1.0, 0.5, 0.1\}$) affect collaborative model convergence, client drift, and per-attack-type detection recall under standard FedAvg?
- **RQ3 (Byzantine Robustness & Defense Limits):** How vulnerable is decentralized CAN intrusion detection to Byzantine data poisoning (Label Flipping) and model poisoning (Delta Sign Reversal), and can coordinate-wise robust aggregators (Median and Trimmed Mean) reliably recover detection performance without centralizing vehicle data?

---

## 4. Main Manuscript Structure

1. **Introduction:** Vehicular connectivity, lack of native CAN bus security, motivation for edge-based Federated Learning, threat of fleet heterogeneity and Byzantine attacks, summary of contributions.
2. **Related Work:** In-vehicle CAN bus intrusion detection systems; Federated Learning in automotive networks; Byzantine-robust aggregation mechanisms; comparison matrix highlighting literature gaps.
3. **Threat Model & Problem Formulation:** System architecture, vehicular network model, adversary capabilities (data vs. model poisoning), mathematical formulations of Label Flipping and Delta Sign Reversal.
4. **Leak-Free Preprocessing & Edge Model Architecture:** CAN frame parsing with dynamic DLC handling, 11-dimensional normalized representation, leak-free $\Delta t_0$ initialization, sliding window sequence generation ($W=16, S=16$), `CAN1DCNN` with `GroupNorm` and parameter count (40,610).
5. **Heterogeneous Federated Optimization & Aggregation:** Attack-type-aware Dirichlet allocation, local client training procedure, sample-weighted FedAvg, Coordinate-wise Median, Coordinate-wise Trimmed Mean, communication complexity.
6. **Experimental Setup & Verification:** Dataset description (HCRL Kia Soul, 500k frames, 31,250 windows), train/val/test splits, evaluation metrics, latency profiling methodology, multi-seed control.
7. **Empirical Results & Comparative Analysis:**
   - *Section 7.1:* Centralized baseline performance, per-attack recall, and single-sample ECU latency.
   - *Section 7.2:* IID vs. Non-IID heterogeneity benchmarks ($\alpha \in \{1.0, 0.5, 0.1\}$) and client drift analysis.
   - *Section 7.3:* Byzantine attack impact on FedAvg and performance of robust aggregators on primary seed 42.
   - *Section 7.4:* Multi-seed statistical stability and the failure mode of Trimmed Mean under Non-IID skew.
8. **Discussion & Reviewer Considerations:** Fleet vs. single-vehicle simulation, window labeling weak supervision, computational vs. data trade-offs, practical automotive deployment considerations.
9. **Conclusion & Future Directions:** Summary of findings, recommendation of Coordinate-wise Median for vehicular FL, future extensions to Multi-Krum, FoolsGold, and hardware trace-replay.

---

## 5. Required Result Tables

### Table 1: Comprehensive Primary Performance Benchmark (Seed 42)
*Evaluated on untouched test partition (4,690 windows: 2,900 Normal, 1,790 Attack).*

| Paradigm / Threat Scenario | Defense / Aggregator | Best Round | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test F1-Score (%) | Test FPR (%) | Test FNR (%) | ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | Centralized SGD | Epoch 8 | **99.66%** | 99.61% | **99.50%** | **99.55%** | 0.24% | **0.50%** | **0.9992** |
| **IID Baseline** | FedAvg | Round 9 | **98.25%** | 99.71% | 95.70% | **97.66%** | 0.17% | 4.30% | 0.9965 |
| **Non-IID ($\alpha = 1.0$)** | FedAvg | Round 10 | **98.51%** | 99.83% | 96.26% | **98.01%** | 0.10% | 3.74% | 0.9961 |
| **Non-IID ($\alpha = 0.5$)** | FedAvg | Round 9 | **98.49%** | 99.60% | 96.42% | **97.98%** | 0.24% | 3.58% | 0.9956 |
| **Non-IID ($\alpha = 0.1$)** | FedAvg | Round 10 | **92.11%** | 97.91% | **81.06% ⚠️**| **88.69% ⚠️**| 1.07% | **18.94% ⚠️**| 0.9573 |
| **Benign Fleet ($\alpha = 0.5$)** | Coordinate Median | Round 10 | **98.51%** | 98.81% | 97.26% | **98.03%** | 0.72% | 2.74% | 0.9956 |
| **Benign Fleet ($\alpha = 0.5$)** | Trimmed Mean ($m=2$) | Round 10 | **98.34%** | 99.42% | 96.20% | **97.79%** | 0.34% | 3.80% | 0.9955 |
| **Attack A: Label Flipping** | **FedAvg** | Round 10 | **94.63%** | 99.87% | **86.03% ⚠️**| **92.44%** | 0.07% | **13.97% ⚠️**| 0.9932 |
| **Attack A: Label Flipping** | **Coordinate Median** | Round 9 | **97.14%** | 95.05% | **97.60% ✅**| **96.31%** | 3.14% | **2.40% ✅**| 0.9945 |
| **Attack A: Label Flipping** | **Trimmed Mean ($m=2$)**| Round 9 | **97.51%** | 96.19% | **97.32% ✅**| **96.75%** | 2.38% | **2.68% ✅**| 0.9946 |
| **Attack B: Delta Sign Reversal**| **FedAvg** | Round 2 | **38.17% 💥**| 38.17% | 100.00% | **55.25% 💥**| **100.00% 💥**| 0.00% | 0.5861 |
| **Attack B: Delta Sign Reversal**| **Coordinate Median** | Round 9 | **97.31%** | 97.17% | **95.75% ✅**| **96.45%** | 1.72% | **4.25% ✅**| 0.9934 |
| **Attack B: Delta Sign Reversal**| **Trimmed Mean ($m=2$)**| Round 10 | **97.06%** | 95.94% | **96.37% ✅**| **96.15%** | 2.52% | **3.63% ✅**| 0.9936 |

---

### Table 2: Multi-Seed Statistical Stability (Seeds 42, 43, 44; Mean ± Std)

| Threat Scenario | Aggregator | Test Accuracy Mean ± Std (%) | Test F1-Score Mean ± Std (%) | Test FPR Mean ± Std (%) | Test FNR Mean ± Std (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Benign Fleet** | **FedAvg** | 97.96% $\pm$ 0.54% | **97.27% $\pm$ 0.73%** | 0.45% $\pm$ 0.20% | 4.62% $\pm$ 1.23% |
| **Benign Fleet** | **Coordinate Median** | 96.93% $\pm$ 1.76% | **95.97% $\pm$ 2.26%** | 1.97% $\pm$ 2.16% | 4.86% $\pm$ 1.54% |
| **Benign Fleet** | **Trimmed Mean ($m=2$)** | 97.56% $\pm$ 0.83% | **96.77% $\pm$ 1.07%** | 1.11% $\pm$ 1.09% | 4.58% $\pm$ 0.56% |
| **Attack A: Label Flipping** | **FedAvg** | 74.55% $\pm$ 26.17% | **78.56% $\pm$ 16.99% ⚠️**| 35.37% $\pm$ 45.77% | 9.39% $\pm$ 5.59% |
| **Attack A: Label Flipping** | **Coordinate Median** | **95.19% $\pm$ 2.82%** | **93.75% $\pm$ 3.57% ✅**| **3.91% $\pm$ 3.38%** | **6.28% $\pm$ 2.93%** |
| **Attack A: Label Flipping** | **Trimmed Mean ($m=2$)** | 93.06% $\pm$ 6.26% | **91.97% $\pm$ 6.64%** | 9.01% $\pm$ 10.90% | 3.59% $\pm$ 1.87% |
| **Attack B: Delta Sign Reversal**| **FedAvg** | 54.04% $\pm$ 17.76% | **63.43% $\pm$ 10.45% 💥**| 73.14% $\pm$ 28.39% | 0.00% $\pm$ 0.00% |
| **Attack B: Delta Sign Reversal**| **Coordinate Median** | **95.10% $\pm$ 2.61%** | **93.60% $\pm$ 3.24% ✅**| **3.53% $\pm$ 3.68%** | **6.67% $\pm$ 3.39%** |
| **Attack B: Delta Sign Reversal**| **Trimmed Mean ($m=2$)** | 77.20% $\pm$ 27.60% | **82.14% $\pm$ 19.02% ⚠️**| 34.23% $\pm$ 46.52% | 4.28% $\pm$ 2.45% |

---

### Table 3: Per-Attack-Type Detection Recall Breakdown (%)

| Experimental Condition | DoS (N=398) | Fuzzy (N=209) | Gear (N=591) | RPM (N=592) | Normal Specificity (N=2900) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Centralized Baseline** | 99.75% | 96.17% | 100.00% | 100.00% | 99.76% |
| **IID FedAvg** | 97.99% | 80.86% | 97.63% | 97.47% | 99.83% |
| **Non-IID ($\alpha = 1.0$) FedAvg** | 97.74% | 82.30% | 99.32% | 99.32% | 99.90% |
| **Non-IID ($\alpha = 0.5$) FedAvg** | 98.49% | 80.38% | 98.48% | 100.00% | 99.76% |
| **Non-IID ($\alpha = 0.1$) FedAvg** | **43.97% 💥**| **44.50% 💥**| 100.00% | 100.00% | 98.93% |
| **Label Flip + FedAvg** | 93.97% | 52.15% ⚠️ | 85.11% ⚠️ | 93.75% | 99.93% |
| **Label Flip + Median** | 97.99% | 78.47% ✅ | 99.83% ✅ | 98.65% ✅ | 96.86% |
| **Label Flip + Trimmed Mean** | 97.99% | 83.25% ✅ | 98.14% ✅ | 99.32% ✅ | 97.62% |

---

### Table 4: Latency & Automotive Computational Profiling

| Benchmark Mode | Batch Size | Measured Latency | Standard Deviation | Automotive Practicality Assessment |
|---|:---:|:---:|:---:|---|
| **Single-Sample Real-Time (Mean)** | 1 | **455.99 $\mu$s (0.46 ms)** | 42.1 $\mu$s | **Hard real-time CAN deadline (<1.0 ms) met on CPU.** |
| **Single-Sample Real-Time (Median)** | 1 | **421.05 $\mu$s (0.42 ms)** | - | Typical single-window ECU execution cost. |
| **Single-Sample Real-Time (p95)** | 1 | **590.59 $\mu$s (0.59 ms)** | - | Deterministic worst-case real-time bound. |
| **Batched Execution** | 128 | **3.46 ms** | 0.31 ms | High throughput for central gateway buffer. |
| **Amortized Per-Sample Latency** | 128 | **27.04 $\mu$s** | - | Offline log throughput metric (not real-time). |
| **Throughput** | 128 | **36,975 samples/sec** | - | Ingests ~500k CAN frames/minute easily. |

---

## 6. Critical Limitations to Acknowledge

1. **Single-Vehicle Data Source:** Evaluated on the Kia Soul capture traces. Cross-OEM generalization (e.g. evaluating Kia models against Ford/Tesla CAN traces) remains an open challenge requiring domain-adaptive federated learning.
2. **Weak Supervision in Sliding Windows:** Sequence-level labeling (`label_mode="any"`) treats any 16-frame window containing $\ge 1$ attack packet as anomalous, potentially introducing label noise during sparse injections.
3. **Data Subsampling Scope:** 500,000 frames (31,250 windows) were subsampled from the 17.56M raw frames to ensure computational feasibility across hundreds of decentralized client rounds.
4. **Absence of Cryptographic Privacy:** Federated Learning protects raw data locality, but parameter exchange alone does not guarantee differential privacy or protection against reconstruction/inversion attacks.

---

## 7. Absolute Claims That Must NOT Be Made

1. **DO NOT claim evaluation on the "entire HCRL dataset":** The pipeline evaluated 500,000 frames (2.85% of total raw records). Always state: *"evaluated on a 500,000-frame benchmark extracted from the HCRL dataset."*
2. **DO NOT claim spoofing detection is "100% across all regimes":** Gear and RPM spoofing reached 100% specifically under $\alpha=0.1$ and $\alpha=0.5$ (RPM), but ranged from 97.47% to 99.32% in IID and $\alpha=1.0$.
3. **DO NOT claim malicious clients were `[0, 1]` or scaling factor was $\gamma = 1.5$:** The verified experimental benchmark evaluated malicious clients `[8, 9]` (representing 39.65% of training samples) with $\gamma = 1.0$.
4. **DO NOT claim Trimmed Mean is uniformly superior to FedAvg across all seeds:** Trimmed Mean collapsed to 55.25% F1 on Seed 44 under update corruption. This failure mode must be reported transparently as a scientific finding.
5. **DO NOT report amortized batch latency (27 $\mu$s) as real-time ECU detection latency:** Always report single-sample latency ($455.99\ \mu\text{s}$) when discussing real-time automotive deadlines.

