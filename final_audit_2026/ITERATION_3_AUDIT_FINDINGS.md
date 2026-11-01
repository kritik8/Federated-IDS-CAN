# ITERATION 3 AUDIT FINDINGS: SCIENTIFIC AND TECHNICAL RECONCILIATION
**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Date:** October 31, 2026  
**Status:** Implementation Freeze & Validation Gate (Iteration 3)

---

## 1. Executive Summary

This audit report records the comprehensive investigation of experimental artifacts, configurations, scripts, and metric summaries following Iterations 1 and 2. 

Every inconsistency identified between the previous chat summaries, written reports, and underlying raw artifacts has been systematically traced to its origin. Crucially, the raw experimental artifacts (`results/metrics/*.json`, `results/*_multiseed/`) remain intact, uncorrupted, and mathematically sound. Discrepancies have been resolved by cross-checking raw confusion matrices, execution logs, and script code.

---

## 2. Inventory of Identified Inconsistencies & Root Cause Analysis

### Finding 1: Coordinate-wise Median Identical Values in Conversational Summary
- **Symptom:** In the assistant conversational response table summarizing Iteration 2, Coordinate-wise Median reported identical numbers for both Benign Clean and Attack A (Label Flipping) across Accuracy ($94.61\% \pm 1.63\%$), F1 ($93.75\% \pm 2.29\%$), FPR ($3.60\% \pm 2.06\%$), and FNR ($7.39\% \pm 5.34\%$).
- **Investigation of Raw Artifacts:**
  - File inspected: `results/metrics/robust_multiseed_summary.json` and `results/metrics/robust_all_aggregators_summary.csv`.
  - The actual saved results are genuinely distinct and mathematically verified:
    - **Benign Clean Median:** Accuracy = $96.93\% \pm 1.76\%$, F1 = $95.97\% \pm 2.26\%$, FPR = $1.97\% \pm 2.16\%$, FNR = $4.86\% \pm 1.54\%$.
    - **Attack A (Label Flip) Median:** Accuracy = $95.19\% \pm 2.82\%$, F1 = $93.75\% \pm 3.57\%$, FPR = $3.91\% \pm 3.38\%$, FNR = $6.28\% \pm 2.93\%$.
    - **Attack B (Sign Reversal) Median:** Accuracy = $95.10\% \pm 2.61\%$, F1 = $93.60\% \pm 3.24\%$, FPR = $3.53\% \pm 3.68\%$, FNR = $7.13\% \pm 2.04\%$.
- **Root Cause:** A formatting copy-paste error occurred exclusively in the assistant's Markdown chat response during the turn summary. The underlying raw JSON and CSV artifact files (`results/metrics/robust_all_aggregators_summary.csv`) were never corrupted and contained the correct, distinct metrics.

---

### Finding 2: FedAvg Attack B Metric Inconsistency
- **Symptom:** The assistant conversational summary reported FedAvg under Attack B as:
  `Accuracy: 56.33%, Precision: 0%, Recall: 0%, F1: 0%, FPR: 0%, FNR: 100%`.
  Under the held-out test set of 4,690 windows (2,900 Normal, 1,790 Attack), an all-negative classifier would achieve an accuracy of $2,900 / 4,690 = 61.83\%$, not $56.33\%$.
- **Investigation of Raw Artifacts:**
  - File inspected: `results/metrics/robust_multiseed_summary.json` (key `update_corrupt_fedavg`) and `results/metrics/robust_all_aggregators_summary.csv`.
  - The actual multi-seed metrics across Seeds 42, 43, 44 are:
    - **F1-Score:** $63.43\% \pm 10.45\%$
    - **Accuracy:** $54.04\% \pm 17.76\%$
    - **FPR:** $73.14\% \pm 28.39\%$
    - **FNR:** $1.94\% \pm 2.32\%$
  - Inspection of raw Seed 42 confusion matrix:
    - Confusion Matrix: `[[0, 2900], [0, 1790]]` (TN=0, FP=2900, FN=0, TP=1790).
    - Accuracy: $1790 / 4690 = 38.166\%$
    - Precision: $1790 / 4690 = 38.166\%$
    - Recall: $1790 / 1790 = 100.0\%$
    - F1-Score: $2 \times (0.38166 \times 1.0) / (0.38166 + 1.0) = 55.2469\%$ ($55.25\%$)
    - FPR: $2900 / 2900 = 100.0\%$
    - FNR: $0 / 1790 = 0.0\%$
- **Root Cause:** In the chat summary, the assistant mistakenly described FedAvg as failing via "all-negative" predictions with hallucinated numbers (56.33%), whereas the actual model collapsed towards predicting **all-positive** (attack) under Seed 42 and Seed 44 due to the sign-reversed gradient updates pushing the classifier bias towards positive logits. The actual raw CSV/JSON records the correct multi-seed mean of $63.43\%$ F1 and $54.04\%$ Accuracy.

---

### Finding 3: Multi-Krum Seed 44 Behavior and Client Exclusion Dynamics
- **Symptom:** The Iteration 2 narrative stated that Multi-Krum collapsed to F1 = $55.25\%$ across all three regimes (Benign, Attack A, Attack B) on Seed 44 due to selecting Byzantine clients.
- **Investigation of Raw Diagnostics:**
  - File inspected: `results/metrics/robust_multikrum_experiments.json`.
  - Multi-seed results for Seed 44:
    - **Benign:** Accuracy = $87.40\%$, Precision = $98.00\%$, Recall = $68.38\%$, F1 = $80.55\%$. Confusion matrix: `[[2875, 25], [566, 1224]]`. It did **not** collapse to 55.25% in Benign mode!
    - **Attack A (Label Flip):** Accuracy = $38.17\%$, Precision = $38.17\%$, Recall = $100.0\%$, F1 = $55.25\%$. Confusion matrix: `[[0, 2900], [0, 1790]]`.
    - **Attack B (Sign Reversal):** Accuracy = $38.17\%$, Precision = $38.17\%$, Recall = $100.0\%$, F1 = $55.25\%$. Confusion matrix: `[[0, 2900], [0, 1790]]`.
  - Client partition and Krum selection analysis:
    - Malicious client IDs: `{8, 9}`.
    - Selected clients across rounds 1 to 10 in Attack A and Attack B: **`{0, 1, 2, 4, 5, 7}`** in every single round!
    - Malicious clients 8 and 9 were **never selected** in any round.
    - Excluded clients in every round: **`{3, 6, 8, 9}`**.
    - Inspection of local class distributions under Dirichlet $\alpha=0.5$ (Seed 44):
      - Clients 0, 1, 2, 4, 5, 7 hold mostly attack traffic (83.5% to 100% Attack samples; Client 2 has 100% attack samples).
      - Client 3 holds 5,159 samples (96.9% Normal traffic; 4,998 Normal samples, which is 37.5% of all normal training data).
      - Because clients {0, 1, 2, 4, 5, 7} all trained on attack-dominated batches, their parameter updates clustered closely in $\mathbb{R}^{40,610}$.
      - Multi-Krum calculated pairwise Euclidean distances, found the {0, 1, 2, 4, 5, 7} cluster to have the lowest distance sum, and pruned honest Client 3 as a distance "outlier".
      - Consequently, the aggregated global model received almost no normal training samples and learned to predict all frames as Attack!
- **Scientific Significance:** Multi-Krum's failure on Seed 44 is not caused by admitting Byzantine clients, but by **geometric pruning of legitimate Non-IID minority clients**. Distance-based aggregators equate statistical heterogeneity with adversarial poisoning.

---

### Finding 4: Centralized CNN vs. Random Forest Dual Profile Discrepancy
- **Symptom:** Two different CNN performance profiles appeared in discussion:
  - Profile 1: Accuracy $99.65\% \pm 0.10\%$, F1 $99.54\% \pm 0.13\%$, 10 epochs, 28.91 s.
  - Profile 2: Accuracy $97.75\% \pm 0.12\%$, F1 $97.47\% \pm 0.13\%$, 15 epochs, 103.88 s.
- **Investigation of Canonical Baseline:**
  - Files inspected: `results/multiseed_summary_baseline.json`, `results/metrics/baseline_results.json`, and `results/metrics/centralized_cnn_vs_rf_comparison.csv`.
  - The canonical centralized 1D-CNN baseline established in Iteration 1 across Seeds 42, 43, 44 has:
    - Seed 42: Accuracy = $99.66\%$, F1 = $99.55\%$
    - Seed 43: Accuracy = $99.55\%$, F1 = $99.41\%$
    - Seed 44: Accuracy = $99.74\%$, F1 = $99.66\%$
    - **Mean $\pm$ Std:** Accuracy = $99.65\% \pm 0.10\%$, F1 = $99.54\% \pm 0.13\%$, Epochs = 10, Duration = $28.91 \pm 0.94$ s.
  - Profile 2 (97.75% Acc, 15 epochs, 103.88s) was an accidental text hallucination from an uncommitted draft test that appeared only in the chat message, not in any committed artifact.
  - `results/metrics/centralized_cnn_vs_rf_comparison.csv` already correctly uses Profile 1.

---

### Finding 5: Real-Time Automotive Feasibility & Latency Claims
- **Symptom:** Previous text asserted that the 1D-CNN "strictly satisfies the $<1$ ms real-time CAN bus deadline" and that Random Forest violates it.
- **Audit Assessment:**
  - Inference was benchmarked on an AMD64 desktop CPU (`AMD64 Family 23 Model 160 Stepping 0`, 8 cores, Windows 11) using PyTorch and scikit-learn in Python 3.12.
  - In a real vehicle, intrusion detection runs on automotive microcontrollers (e.g. ARM Cortex-R/M or AURIX TriCore) or automotive gateways over CAN transceivers.
  - Single-sample Random Forest latency was measured with `n_jobs=-1`, which incurs Windows multi-threading dispatch overhead for a single vector.
  - **Correction:** Remove claims of "automotive safety certification", "ECU hardware validation", or "hard real-time guarantees". Rephrase as measured single-sample inference latency under desktop CPU benchmark conditions, highlighting the relative computational efficiency.

---

### Finding 6: Model Size Basis Standardization
- **Symptom:** The report claimed the 1D-CNN is "28.2× smaller" by comparing CNN FP32 parameter memory ($162.44$ KB) with serialized Random Forest disk size ($4,588.75$ KB).
- **Audit Assessment:** Comparing in-memory parameter tensors to serialized pickle/joblib archives on disk is an apples-to-oranges comparison.
- **Correction:** Report both serialized disk checkpoint sizes and in-memory parameter footprints using consistent criteria.

---

### Finding 7: Single-Vehicle Partition Disclosure & Sliding Window Label Policy
- **Audit Assessment:**
  - The HCRL Car-Hacking dataset was collected from a single Kia Soul vehicle.
  - The 10 federated clients are Dirichlet Non-IID partitions of this single stream, simulating a fleet of 10 virtual vehicle ECUs.
  - The sliding window labelling policy marks a 16-frame window as Attack ($y=1$) if $\ge 1$ frame within the window is an attack frame (any-frame-positive).
- **Correction:** Explicitly disclose both aspects in the README, final audit reports, and figure notes to ensure scientific transparency.

