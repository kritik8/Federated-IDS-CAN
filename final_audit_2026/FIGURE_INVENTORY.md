# Repository Figure & Diagram Inventory: Publication Readiness Audit

**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Audit Date:** October 31, 2026  
**Target Submission Deadline:** November 12, 2026  
**Auditor:** Senior Research Systems Auditor & Academic Reviewer  
**Output Location:** `final_audit_2026/FIGURE_INVENTORY.md`

---

## 1. Inventory of Existing Figures

All 11 existing figure artifacts in `results/figures/` were independently inspected for visual clarity, axis labels, legends, units, resolution (DPI), and alignment with saved experiment JSON metrics.

| # | Existing Filename | Dimensions & Size | Generating Source | What It Demonstrates | Publication Readiness | Action Required |
|---|---|---|---|---|---|---|
| **1** | `baseline_confusion_matrix.png` | 1800 $\times$ 1500 (300 DPI)<br>103.7 KB | `notebooks/03_baseline_ids.ipynb` | Confusion matrix of Centralized 1D-CNN baseline on test set (2893 TN, 7 FP, 9 FN, 1781 TP). | **Ready** | None. Matches `centralized_baseline_v2.json` bit-for-bit. |
| **2** | `baseline_training_curves.png` | 4200 $\times$ 1500 (300 DPI)<br>211.9 KB | `notebooks/03_baseline_ids.ipynb` | 2-panel plot: Cross-Entropy Loss and F1-score across 10 epochs (Train vs. Val). | **Ready** | None. Highlights best checkpoint at Epoch 8. |
| **3** | `eda_can_id_frequency.png` | 3600 $\times$ 1800 (300 DPI)<br>300.9 KB | `notebooks/01_dataset_analysis.ipynb` | Histogram of top 25 arbitration CAN IDs across DoS, Fuzzy, Gear, RPM, Normal. | **Ready** | Suitable for Dataset & Threat Profiling section. |
| **4** | `eda_class_distribution.png` | 2400 $\times$ 1500 (300 DPI)<br>91.3 KB | `notebooks/01_dataset_analysis.ipynb` | Bar chart showing class balance (Normal vs. Attack) across the 5 raw scenarios. | **Ready** | Clear visualization of class distribution. |
| **5** | `eda_temporal_delta_t.png` | 3600 $\times$ 1800 (300 DPI)<br>183.7 KB | `notebooks/01_dataset_analysis.ipynb` | Empirical CDF and log-distribution of inter-arrival times ($\Delta t$) for attacks vs normal. | **Ready** | Crucial for justifying $\Delta t$ feature engineering. |
| **6** | `federated_iid_confusion_matrix.png` | 1800 $\times$ 1500 (300 DPI)<br>102.4 KB | `notebooks/04_federated_iid.ipynb` | Confusion matrix of 10-client IID FedAvg on test set (2895 TN, 5 FP, 77 FN, 1713 TP). | **Ready** | Matches `federated_iid_fedavg.json` exactly. |
| **7** | `federated_iid_learning_curve.png` | 4200 $\times$ 1500 (300 DPI)<br>299.3 KB | `notebooks/04_federated_iid.ipynb` | 2-panel plot: Global Validation Loss and Validation F1 vs. 10 communication rounds. | **Ready** | Highlights convergence and best round at Round 9. |
| **8** | `noniid_client_distributions.png` | 4800 $\times$ 1500 (300 DPI)<br>163.6 KB | Generated during Phase 2B | Stacked bar chart of sample counts and attack categories across 10 clients for $\alpha \in \{1.0, 0.5, 0.1\}$. | **Needs Minor Polish** | High resolution; script generator should be preserved in `scratch/`. |
| **9** | `noniid_f1_comparison.png` | 4500 $\times$ 1500 (300 DPI)<br>401.1 KB | `notebooks/05_federated_noniid.ipynb` | 2-panel plot: Global Val F1 and Val Loss across rounds comparing IID vs $\alpha \in \{1.0, 0.5, 0.1\}$. | **Ready** | Dramatically demonstrates divergence under $\alpha = 0.1$. |
| **10** | `robust_f1_comparison.png` | 4200 $\times$ 1500 (300 DPI)<br>324.4 KB | `scratch/run_robust_experiments.py` | 2-panel plot: Validation F1 convergence comparing FedAvg, Median, Trimmed Mean under Attacks A & B. | **Ready** | Primary visual evidence of Byzantine defense efficacy. |
| **11** | `robust_per_attack_recall.png` | 3000 $\times$ 1500 (300 DPI)<br>132.2 KB | `scratch/run_robust_experiments.py` | Grouped bar chart: Detection rate across DoS, Fuzzy, Gear, RPM under Label Flipping. | **Partially Complete** | Only covers Attack A. Needs companion for Attack B. |

---

## 2. Recommended Figures for Paper Submission

To meet the standards of a top-tier peer-reviewed academic conference (e.g., IEEE T-IFS, IEEE VTC, ACM CCS, or IEEE INFOCOM), the paper should include a curated set of 6 primary figures:

### Figure 1: End-to-End System Architecture & Threat Workflow (NEW DIAGRAM)
- **Status:** MISSING (Must be created).
- **Target Location:** Section 3 (System & Threat Model).
- **Content:** Block diagram illustrating:
  1. Kia Soul in-vehicle CAN bus (high-speed 500 kbps).
  2. Data preprocessing & leak-free feature extraction ($\text{CAN\_ID}, \text{DLC}, D_0\dots D_7, \Delta t$).
  3. Non-overlapping sliding windowing ($W=16$).
  4. Heterogeneous vehicular client fleet partitioned via Dirichlet allocation ($\alpha = 0.5$).
  5. Adversarial clients launching Label Flipping ($y \mapsto 1-y$) and Model Update Corruption ($\Delta' = -\gamma \Delta$).
  6. Server-side robust aggregation (Coordinate-wise Median / Trimmed Mean vs. FedAvg).

### Figure 2: Traffic Characteristics & Feature Dispersions (EXISTING)
- **Status:** READY.
- **Components:** Combine `eda_temporal_delta_t.png` and `eda_can_id_frequency.png` into a unified 2-panel figure.
- **Target Location:** Section 4 (Dataset & Feature Representation).
- **What it Demonstrates:** Shows why inter-arrival time ($\Delta t$) distinguishes DoS attacks (0.3 ms periodic bursts) and why CAN ID entropy separates Fuzzy attacks.

### Figure 3: Non-IID Fleet Heterogeneity & Client Skew (EXISTING)
- **Status:** READY (`results/figures/noniid_client_distributions.png`).
- **Target Location:** Section 5 (Non-IID Partitioning & Federated Baselines).
- **What it Demonstrates:** Visualizes client-level class and category imbalance for $\alpha=1.0$, $\alpha=0.5$, and $\alpha=0.1$, validating the controlled Non-IID experimental design.

### Figure 4: Convergence Dynamics under Fleet Heterogeneity (EXISTING)
- **Status:** READY (`results/figures/noniid_f1_comparison.png`).
- **Target Location:** Section 5 (Empirical Evaluation: Non-IID Performance).
- **What it Demonstrates:** Illustrates that GroupNorm stabilizes FedAvg under moderate and high skew ($\alpha=1.0, 0.5$), but severe client drift induces degradation under extreme skew ($\alpha=0.1$).

### Figure 5: Byzantine Defense Dynamics under Adversarial Attacks (EXISTING)
- **Status:** READY (`results/figures/robust_f1_comparison.png`).
- **Target Location:** Section 6 (Robust Aggregation under Byzantine Attacks).
- **What it Demonstrates:** Proves that Coordinate-wise Median and Trimmed Mean successfully resist both data poisoning (Label Flipping) and model poisoning (Sign Reversal), whereas FedAvg collapses.

### Figure 6: Per-Cyber-Threat Detection Breakdown under Attack (NEW / EXTENDED)
- **Status:** PARTIALLY COMPLETE (Extend `robust_per_attack_recall.png`).
- **Target Location:** Section 6 (Detailed Threat Resilience Breakdown).
- **What it Demonstrates:** 2-panel grouped bar chart showing recall across DoS, Fuzzy, Gear, and RPM for both Label Flipping (Panel A) and Model-Update Corruption (Panel B).

---

## 3. Missing Diagram Specifications (To Be Generated During Paper Preparation)

Before drafting the camera-ready manuscript, the following 4 conceptual and architectural diagrams should be prepared in vector format (PDF/SVG/TikZ):

1. **System & Threat Model Diagram:**
   - Visualizing vehicular CAN bus, edge ECU inference, wireless telematics upload, server aggregation, and adversarial client poisoning.
2. **11-Dimensional Feature Transformation Pipeline:**
   - Visualizing variable DLC parsing, payload normalization, leak-free $\Delta t_0$ initialization, and sliding window concatenation into $(16, 11)$ tensors.
3. **Attack-Type-Aware Dirichlet Allocation Workflow:**
   - Visualizing how 5 raw scenario categories are partitioned across $K=10$ clients and mapped to binary targets while retaining heterogeneous exposure.
4. **Coordinate-wise Aggregation Geometric Intuition:**
   - Visualizing coordinate sorting, median selection, and trimming boundaries ($m=2$) to illustrate why Trimmed Mean can mistakenly trim legitimate client updates under extreme Non-IID client drift.

