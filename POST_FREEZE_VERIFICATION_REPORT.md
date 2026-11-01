# POST-FREEZE VERIFICATION REPORT: READ-ONLY AUDIT & DOCUMENTATION ERRATA

**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Verification Date:** November 1, 2026  
**Implementation Freeze Commit:** `6df5485511a7e88605ff8930b3fece432330d06c`  
**Implementation Freeze Tag:** `implementation-freeze-2026-11-01`  
**Deliverable Path:** `POST_FREEZE_VERIFICATION_REPORT.md` (and `final_audit_2026/POST_FREEZE_VERIFICATION_REPORT.md`)

---

## 1. Frozen Git State Verification

The frozen Git state was rigorously checked against the repository HEAD and the annotated release tag:

### A. Tag and Commit Identity
- **Annotated Tag Object:** `7d673bb71c15c4e75f3d8179356bd0500bbf2265`
- **Target Commit Hash:** `6df5485511a7e88605ff8930b3fece432330d06c`
- **Tag Verification (`git rev-list -n 1 implementation-freeze-2026-11-01`):** `6df5485511a7e88605ff8930b3fece432330d06c`
- **Head Commit (`git rev-parse HEAD`):** `6df5485511a7e88605ff8930b3fece432330d06c`
- **Tag Header (`git show implementation-freeze-2026-11-01 --no-patch`):**
  ```text
  tag implementation-freeze-2026-11-01
  Tagger: Kritik Jain <150144010+kritik8@users.noreply.github.com>
  Date:   Sun Nov 1 11:02:53 2026 +0530

  Official implementation freeze for academic submission (November 1, 2026)

  commit 6df5485511a7e88605ff8930b3fece432330d06c
  Author: Kritik Jain <150144010+kritik8@users.noreply.github.com>
  Date:   Sun Nov 1 11:01:40 2026 +0530

      chore(research): finalize numerical validation and freeze implementation
  ```

### B. Summary of Changes in Freeze Commit (`git show --stat --oneline HEAD`)
- **Total Files Changed:** 92 files (33,292 insertions, 191 deletions)
- **Scope:** Complete multi-seed metrics (`results/*_multiseed/`), verified summary tables (`results/metrics/`), publication figures (`results/figures/`), Random Forest config, Multi-Krum implementation, regression invariants, and comprehensive audit reports (`final_audit_2026/`).

---

## 2. Post-Commit Test Suite Certification

The full automated test suite was executed against the frozen commit using `python -m pytest tests`:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Kratik\IIIT Academics\Hustle\Major_Project_CAN_IDS
plugins: anyio-4.4.0
collected 53 items

tests\test_federated.py ................                                 [ 30%]
tests\test_loader.py ...                                                 [ 35%]
tests\test_metrics.py ...                                                [ 41%]
tests\test_models.py ...                                                 [ 47%]
tests\test_multi_krum.py ........                                        [ 62%]
tests\test_preprocessing.py ...                                          [ 67%]
tests\test_robust_aggregation.py ..........                              [ 86%]
tests\test_validation_invariants.py .......                              [100%]

============================= 53 passed in 46.06s =============================
```

- **Tests Passed:** 53 / 53 (100%)
- **Tests Failed:** 0
- **Warnings:** 0
- **Execution Time:** 46.06 seconds

---

## 3. Latency Documentation Inconsistencies & Root Cause Analysis

A forensic investigation of `results/metrics/latency_and_size_audit.json` and `scratch/audit_benchmarks_and_metrics.py` resolved the latency discrepancies:

### A. Reported Values Investigation
1. **RF Model-Only (Single-Thread):** $30,440.80\ \mu\text{s}$ ($30.44\text{ ms}$)
   - *Status:* **Verified empirical measurement.** Recorded in `latency_and_size_audit.json` from a 200-iteration timed loop with 50 warmup passes.
2. **RF End-to-End (Single-Thread):** $27,666.11\ \mu\text{s}$ ($27.67\text{ ms}$) in detailed latency report
   - *Status:* **Measured artifact, but contaminated by inter-loop variance.** Recorded in `latency_and_size_audit.json` from a separate 200-iteration timed loop. Because single-sample Random Forest inference has large runtime variance (standard deviation $\pm 8,265.72\ \mu\text{s}$), the separate loop averaged lower than the model-only loop.
3. **RF End-to-End (Single-Thread):** $30,649.41\ \mu\text{s}$ ($30.65\text{ ms}$) in final freeze report
   - *Status:* **Documentation synthetic construct.** In earlier draft notes, the author added $30,440.80\ \mu\text{s} + 208.61\ \mu\text{s} = 30,649.41\ \mu\text{s}$ to prevent end-to-end from appearing faster than model-only. It was never an empirical benchmark output.

### B. Benchmark Iteration Count Discrepancy
- **Investigated Discrepancy:** "50 warmup + 200 timed iterations" vs. "1,000 iterations".
- **Source of Truth (`scratch/audit_benchmarks_and_metrics.py`):**
  - CAN 1D-CNN (Model-Only & E2E): 50 warmup + 200 timed iterations.
  - Random Forest Single-Thread (Model-Only & E2E): 50 warmup + 200 timed iterations.
  - Random Forest Multi-Thread (Model-Only): 30 warmup + 100 timed iterations.
- **Finding:** The "1,000 iterations" statement was a narrative documentation error in earlier conversational notes.

### C. Resolution & Manuscript Policy
- **Model-Only Speedups (Verified & Retained):**
  - **$66.8\times$ speedup** over single-threaded Random Forest ($455.99\ \mu\text{s}$ vs. $30,440.80\ \mu\text{s}$).
  - **$157.7\times$ speedup** over multi-threaded Random Forest ($455.99\ \mu\text{s}$ vs. $71,904.69\ \mu\text{s}$).
- **End-to-End RF Latency (Marked Unverified & Omitted):**
  - Because the measured $27,666.11\ \mu\text{s}$ is distorted by cross-loop timing variance and the $30,649.41\ \mu\text{s}$ figure is an unmeasured arithmetic sum, **End-to-End Random Forest latency is formally marked as UNVERIFIED and omitted from comparative claims in the manuscript**. The paper will strictly compare the rigorously verified model-only inference latencies.

---

## 4. Historical Non-IID Report Provenance Correction

An audit of earlier reports and raw artifacts established the exact origin of every erroneous value in draft tables:

| Regime | Authoritative Multi-Seed Ground Truth | Erroneous Draft Table Figure | Exact Provenance of Erroneous Draft Value |
|---|:---:|:---:|---|
| **$\alpha = 0.5$** | **$97.41\% \pm 0.50\%$ F1** ($98.06\% \pm 0.37\%$ Acc) | $97.27\% \pm 0.90\%$ F1 ($97.96\% \pm 0.66\%$ Acc) | Erroneously copied from **Byzantine-Robust Benign FedAvg** (`benign_fedavg` in `robust_all_aggregators_summary.csv` and `robust_multiseed_summary.json`). |
| **$\alpha = 1.0$** | **$97.57\% \pm 0.81\%$ F1** ($98.19\% \pm 0.59\%$ Acc) | $96.42\% \pm 1.62\%$ F1 ($97.31\% \pm 1.20\%$ Acc) | $96.42\%$ was copied from the historical single-seed Non-IID report (`results/reports/federated_noniid_report.md`), where $96.42\%$ was actually the **Test Recall** of $\alpha=0.5$ Seed 42. |
| **$\alpha = 0.1$** | **$69.16\% \pm 17.86\%$ F1** ($78.71\% \pm 12.20\%$ Acc) | $97.02\% \pm 1.45\%$ F1 ($97.77\% \pm 1.09\%$ Acc) | $97.02\%$ was copied from `results/metrics/federated_noniid_alpha_0.5.json`, where $97.02\%$ was the test F1 of $\alpha=0.5$ Seed 42. |

### Permanent Authoritative Baseline Metrics:
- **$\alpha = 1.0$ (Moderate Skew):** **$97.57\% \pm 0.81\%$ F1** ($98.19\% \pm 0.59\%$ Accuracy)
- **$\alpha = 0.5$ (High Skew):** **$97.41\% \pm 0.50\%$ F1** ($98.06\% \pm 0.37\%$ Accuracy)
- **$\alpha = 0.1$ (Extreme Skew):** **$69.16\% \pm 17.86\%$ F1** ($78.71\% \pm 12.20\%$ Accuracy)

No raw experimental results were modified.

---

## 5. Documentation-Only Scientific Wording Corrections

The following precise documentation clarifications were applied:
1. **Privacy Scope:** Clarified that raw CAN records are not directly exchanged between clients and server; however, local model updates may leak information, and no formal differential privacy guarantee is provided.
2. **Client Partitioning:** Described clients as simulated partitions of one Kia Soul dataset, not actual physical vehicles or validated ECUs.
3. **Metadata Cleanup:** Removed generic invented author-role and target-venue metadata from `FINAL_IMPLEMENTATION_FREEZE_REPORT.md` and related audit files.
4. **RF Latency Status:** Updated `LATENCY_SPEEDUP_RECONCILIATION.md` and `FINAL_IMPLEMENTATION_FREEZE_REPORT.md` to classify end-to-end RF latency as unverified and excluded from paper comparisons.

---

## 6. Git Safety & Final Repository Status

- **Annotated Tag `implementation-freeze-2026-11-01`:** Untouched and firmly pointing to commit `6df5485511a7e88605ff8930b3fece432330d06c`.
- **Remote Push:** Never pushed to remote (`origin/main`).
- **Documentation Errata Commit:** Staged and committed separately as `docs: correct documentation errata and scientific wording post-freeze`.

---

## 7. Explicit Confirmation of Artifact Integrity

1. **Source Code (`src/`):** **ZERO changes.** All modules remain identical to the frozen state.
2. **Model Checkpoints (`results/models/`):** **ZERO changes.** Canonical checkpoints are preserved.
3. **Raw Experimental Metrics (`results/*_multiseed/`, `results/metrics/*.json`):** **ZERO changes.** All per-seed confusion matrices and logs remain intact.
4. **Historical Seed 42 Artifacts:** **ZERO changes.**
5. **Experimental Methodology:** **ZERO changes.** No hyperparameters, architectures, or protocols were modified.

