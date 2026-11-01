# NON-IID EXPERIMENTAL RESULT RECONCILIATION REPORT
**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Date:** November 1, 2026  
**Deliverable Path:** `final_audit_2026/NONIID_RESULT_RECONCILIATION.md`

---

## 1. Executive Summary & The Discrepancy

During the numerical consistency audit of the project, a discrepancy in reported Non-IID performance metrics was identified between earlier Iteration 1 artifacts and the narrative table in Section 7 of `FINAL_RESEARCH_VALIDATION_REPORT.md` (drafted during Iteration 3):

| Dirichlet Skew | Iteration 1 Artifacts & Master CSV | Iteration 3 Narrative Report (Section 7) | Discrepancy Description |
|---|:---:|:---:|---|
| **$\alpha = 1.0$ (Moderate Skew)** | **$97.57\% \pm 0.81\%$ F1** ($98.19\% \pm 0.59\%$ Acc) | $96.42\% \pm 1.62\%$ F1 ($97.31\% \pm 1.20\%$ Acc) | $\Delta \text{F1} = -1.15\%$ |
| **$\alpha = 0.5$ (High Skew)** | **$97.41\% \pm 0.50\%$ F1** ($98.06\% \pm 0.37\%$ Acc) | $97.27\% \pm 0.90\%$ F1 ($97.96\% \pm 0.66\%$ Acc) | $\Delta \text{F1} = -0.14\%$ |
| **$\alpha = 0.1$ (Extreme Skew)** | **$69.16\% \pm 17.86\%$ F1** ($78.71\% \pm 12.20\%$ Acc) | $97.02\% \pm 1.45\%$ F1 ($97.77\% \pm 1.09\%$ Acc) | **$\Delta \text{F1} = +27.86\%$** (Severe) |

The discrepancy under $\alpha = 0.1$ ($\Delta = +27.86\%$) cannot be attributed to variance estimators (sample vs. population standard deviation). A full forensic audit was executed to locate every underlying per-seed artifact, determine the mathematical ground truth, identify the root cause, and establish the permanent authoritative result.

---

## 2. Inventory of Inspected Source Files

The following original files and directories were inspected:

1. **Per-Seed Non-IID Raw JSON Artifacts:**
   - `results/noniid_multiseed/alpha_1.0/seed_42/metrics.json`
   - `results/noniid_multiseed/alpha_1.0/seed_43/metrics.json`
   - `results/noniid_multiseed/alpha_1.0/seed_44/metrics.json`
   - `results/noniid_multiseed/alpha_0.5/seed_42/metrics.json`
   - `results/noniid_multiseed/alpha_0.5/seed_43/metrics.json`
   - `results/noniid_multiseed/alpha_0.5/seed_44/metrics.json`
   - `results/noniid_multiseed/alpha_0.1/seed_42/metrics.json`
   - `results/noniid_multiseed/alpha_0.1/seed_43/metrics.json`
   - `results/noniid_multiseed/alpha_0.1/seed_44/metrics.json`
2. **Canonical Baseline Summary Artifacts:**
   - `results/multiseed_summary_noniid.json`
   - `results/multiseed_summary_noniid.csv`
   - `results/metrics/final_verified_results.csv`
   - `results/metrics/final_verified_results.json`
3. **Single-Seed Benchmark Artifacts:**
   - `results/metrics/federated_noniid_alpha_1.0.json` (Seed 42)
   - `results/metrics/federated_noniid_alpha_0.5.json` (Seed 42)
   - `results/metrics/federated_noniid_alpha_0.1.json` (Seed 42)
4. **Historical Reports:**
   - `results/reports/federated_noniid_report.md`
   - `final_audit_2026/ITERATION_1_REPORT.md`

---

## 3. Forensic Per-Seed Comparison & Arithmetic Verification

All 9 individual runs across the 3 Dirichlet alpha regimes were independently extracted from their raw `metrics.json` files and evaluated against the held-out test partition ($N=4,690$ windows: 2,900 Normal, 1,790 Attack):

### A. Dirichlet $\alpha = 1.0$ (Moderate Skew)

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 98.51% | 99.83% | 96.26% | 98.01% | 0.10% | 3.74% | 0.9961 |
| **43** | 98.55% | 99.20% | 96.98% | 98.08% | 0.48% | 3.02% | 0.9971 |
| **44** | 97.51% | 99.59% | 93.85% | 96.64% | 0.24% | 6.15% | 0.9959 |
| **Mean** | **98.19%** | **99.54%** | **95.70%** | **97.57%** | **0.28%** | **4.30%** | **0.9964** |
| **Std ($ddof=1$)** | **± 0.59%** | **± 0.32%** | **± 1.64%** | **± 0.81%** | **± 0.19%** | **± 1.64%** | **± 0.0007** |

### B. Dirichlet $\alpha = 0.5$ (High Skew)

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 98.49% | 99.60% | 96.42% | 97.98% | 0.24% | 3.58% | 0.9956 |
| **43** | 97.91% | 99.94% | 94.58% | 97.19% | 0.03% | 5.42% | 0.9975 |
| **44** | 97.78% | 98.56% | 95.59% | 97.05% | 0.86% | 4.41% | 0.9958 |
| **Mean** | **98.06%** | **99.37%** | **95.53%** | **97.41%** | **0.38%** | **4.47%** | **0.9963** |
| **Std ($ddof=1$)** | **± 0.37%** | **± 0.72%** | **± 0.92%** | **± 0.50%** | **± 0.43%** | **± 0.92%** | **± 0.0010** |

### C. Dirichlet $\alpha = 0.1$ (Extreme Skew)

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FPR (%) | FNR (%) | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 92.11% | 97.91% | 81.06% | 88.69% | 1.07% | 18.94% | 0.9573 |
| **43** | 75.76% | 99.25% | 36.76% | 53.65% | 0.17% | 63.24% | 0.8766 |
| **44** | 68.25% | 56.07% | 77.71% | 65.14% | 37.59% | 22.29% | 0.7988 |
| **Mean** | **78.71%** | **84.41%** | **65.18%** | **69.16%** | **12.94%** | **34.82%** | **0.8776** |
| **Std ($ddof=1$)** | **± 12.20%** | **± 24.55%** | **± 24.67%** | **± 17.86%** | **± 21.35%** | **± 24.67%** | **± 0.0792** |

---

## 4. Root Cause Analysis

1. **Origin of the Correct Artifacts:**
   The raw metrics in `results/noniid_multiseed/` and the consolidated files `results/multiseed_summary_noniid.json`, `results/multiseed_summary_noniid.csv`, and `results/metrics/final_verified_results.csv` **all agree exactly**. They record:
   - $\alpha=1.0$: $97.57\% \pm 0.81\%$ F1
   - $\alpha=0.5$: $97.41\% \pm 0.50\%$ F1
   - $\alpha=0.1$: $69.16\% \pm 17.86\%$ F1
2. **Origin of the False Values in Earlier Reports (Detailed Provenance):**
   - **For $\alpha = 0.5$ ($97.27\% \pm 0.90\%$ F1):** In the narrative table of Section 7 in draft `FINAL_RESEARCH_VALIDATION_REPORT.md`, the author mistakenly transcribed the **Byzantine-Robust Benign FedAvg** result (`benign_fedavg` in `robust_all_aggregators_summary.csv` and `robust_multiseed_summary.json`: $97.27\% \pm 0.90\%$ F1, $97.96\% \pm 0.66\%$ Acc) into the Non-IID baseline table slot. The true clean Non-IID baseline for $\alpha=0.5$ (from `multiseed_summary_noniid.csv` and `results/noniid_multiseed/alpha_0.5/`) is **$97.41\% \pm 0.50\%$ F1** ($98.06\% \pm 0.37\%$ Acc).
   - **For $\alpha = 1.0$ ($96.42\% \pm 1.62\%$ F1):** The figure $96.42\%$ was taken from the historical single-seed Non-IID report (`results/reports/federated_noniid_report.md`), where $96.42\%$ was actually the **Test Recall** of $\alpha=0.5$ Seed 42, erroneously copied as F1 for $\alpha=1.0$. The true multi-seed result is **$97.57\% \pm 0.81\%$ F1** ($98.19\% \pm 0.59\%$ Acc).
   - **For $\alpha = 0.1$ ($97.02\% \pm 1.45\%$ F1):** The figure $97.02\%$ was copied from `results/metrics/federated_noniid_alpha_0.5.json`, where $97.02\%$ was the validation/test metric for $\alpha=0.5$ Seed 42. This typo obscured the actual multi-seed result: **$69.16\% \pm 17.86\%$ F1** ($78.71\% \pm 12.20\%$ Acc).
3. **Scientific Realism of $\alpha = 0.1$ Collapse:**
   In vehicular networks under extreme Dirichlet heterogeneity ($\alpha = 0.1$), client partitions suffer severe class skew:
   - Certain clients observe only cyber-attacks with zero normal frames, while others observe exclusively normal telemetry.
   - When standard FedAvg averages client weights without proximal regularization (FedProx) or control variates (SCAFFOLD), **severe client drift** occurs.
   - In Seed 43, the global model missed **63.24% of attack windows** (Recall dropped to $36.76\%$, F1 collapsed to $53.65\%$).
   - In Seed 44, the global model suffered an **FPR of 37.59%** (F1 = $65.14\%$).
   - The resulting multi-seed mean of **$69.16\% \pm 17.86\%$** is a foundational scientific finding of this thesis: **Extreme Non-IID CAN traffic induces severe client drift that destabilizes FedAvg.** The claim of $97.02\%$ was a typographical error that obscured this critical vulnerability.

---

## 5. Authoritative Decision & Resolution

- **Authoritative Values Adopted:**
  - **$\alpha = 1.0$:** $\text{F1} = \mathbf{97.57\% \pm 0.81\%}$, $\text{Accuracy} = \mathbf{98.19\% \pm 0.59\%}$, $\text{Precision} = \mathbf{99.54\% \pm 0.32\%}$, $\text{Recall} = \mathbf{95.70\% \pm 1.64\%}$, $\text{FPR} = \mathbf{0.28\% \pm 0.19\%}$, $\text{FNR} = \mathbf{4.30\% \pm 1.64\%}$.
  - **$\alpha = 0.5$:** $\text{F1} = \mathbf{97.41\% \pm 0.50\%}$, $\text{Accuracy} = \mathbf{98.06\% \pm 0.37\%}$, $\text{Precision} = \mathbf{99.37\% \pm 0.72\%}$, $\text{Recall} = \mathbf{95.53\% \pm 0.92\%}$, $\text{FPR} = \mathbf{0.38\% \pm 0.43\%}$, $\text{FNR} = \mathbf{4.47\% \pm 0.92\%}$.
  - **$\alpha = 0.1$:** $\text{F1} = \mathbf{69.16\% \pm 17.86\%}$, $\text{Accuracy} = \mathbf{78.71\% \pm 12.20\%}$, $\text{Precision} = \mathbf{84.41\% \pm 24.55\%}$, $\text{Recall} = \mathbf{65.18\% \pm 24.67\%}$, $\text{FPR} = \mathbf{12.94\% \pm 21.35\%}$, $\text{FNR} = \mathbf{34.82\% \pm 24.67\%}$.
- **Was Any Rerun Necessary?**
  **No.** The original raw experiment files in `results/noniid_multiseed/alpha_{1.0, 0.5, 0.1}/seed_{42, 43, 44}/metrics.json` were complete, intact, uncorrupted, and mathematically verified.
- **Files Corrected:**
  - `final_audit_2026/FINAL_RESEARCH_VALIDATION_REPORT.md` (corrected Section 7 table rows to match `final_verified_results.csv`).
  - `results/metrics/final_validation_checks.json` (added explicit Non-IID numerical integrity check).
