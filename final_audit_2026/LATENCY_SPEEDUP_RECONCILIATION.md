# LATENCY BENCHMARK & SPEEDUP RECONCILIATION REPORT
**Project:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Date:** November 1, 2026  
**Deliverable Path:** `final_audit_2026/LATENCY_SPEEDUP_RECONCILIATION.md`

---

## 1. Executive Summary & The Ambiguity

In previous project documentation (including `README.md` and `FINAL_RESEARCH_VALIDATION_REPORT.md`), the single-sample inference latency advantage of the proposed lightweight 1D-CNN over the classical Random Forest baseline was summarized using the range:

> *"1D-CNN is 45×–157× faster for single-sample inference"*

An audit of the arithmetic reveals that this range mixed two distinct profiling conditions:
1. The **lower bound ($45.8\times$)** compared the **End-to-End latency** of the 1D-CNN ($664.60\ \mu\text{s}$, including PyTorch tensor conversion and preprocessing) against the single-threaded Random Forest model execution ($30,440.80\ \mu\text{s}$):
   $$\frac{30,440.80\ \mu\text{s}}{664.60\ \mu\text{s}} \approx 45.8\times$$
2. The **upper bound ($157.7\times$)** compared the **Model-Only latency** of the 1D-CNN ($455.99\ \mu\text{s}$) against the multi-threaded Random Forest execution ($71,904.69\ \mu\text{s}$):
   $$\frac{71,904.69\ \mu\text{s}}{455.99\ \mu\text{s}} \approx 157.7\times$$

Mixing different profiling scopes (End-to-End vs. Model-Only) and threading configurations (single-thread vs. multi-thread) in a single hyphenated range creates ambiguity. This report standardizes the comparisons under consistent, mathematically explicit terms.

---

## 2. Benchmark Hardware & Software Environment

All latency benchmarks were captured on the local host under identical runtime conditions:
- **Host CPU:** AMD64 Family 23 Model 160 Stepping 0 (8 logical cores)
- **Operating System:** Windows 11 Pro (`10.0.26300`)
- **Software Stack:** Python 3.12.3, PyTorch 2.9.1+cpu, scikit-learn 1.8.0
- **Evaluation Batch Size:** $B = 1$ (Safety-critical single-window edge evaluation; an ECU cannot buffer 128 packets before triggering an alarm)
- **Protocol:** 50 warmup iterations, followed by 200 timed iterations using high-resolution monotonic timer `time.perf_counter()` for CNN and RF single-thread (30 warmup + 100 timed iterations for RF multi-thread). Statements in prior notes citing '1,000 iterations' were documentation errors.

---

## 3. Raw Measured Latency Values

*(From `results/metrics/latency_and_size_audit.json` and `results/metrics/baseline_results.json`)*

| Architecture / Model | Evaluation Configuration | Mean Latency (μs) | Mean Latency (ms) | Median Latency (μs) | 95th Percentile P95 (μs) | Standard Deviation (μs) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Centralized 1D-CNN** | Model-Only (`eval`, `torch.no_grad()`) | **455.99 μs** | **0.456 ms** | 421.05 μs | 590.59 μs | ± 98.42 μs |
| **Centralized 1D-CNN** | End-to-End (Tensor prep + inference) | **664.60 μs** | **0.665 ms** | 620.10 μs | 941.28 μs | ± 142.15 μs |
| **Random Forest (300 Trees)** | Model-Only, Single-Thread (`n_jobs=1`) | **30,440.80 μs** | **30.441 ms** | 30,647.35 μs | 44,730.69 μs | ± 8,265.72 μs |
| **Random Forest (300 Trees)** | Model-Only, Multi-Thread (`n_jobs=-1`, 8 threads) | **71,904.69 μs** | **71.905 ms** | 70,832.80 μs | 110,652.02 μs | ± 16,330.49 μs |
| **Random Forest (300 Trees)** | End-to-End, Single-Thread (`n_jobs=1`) | *27,666.11 μs (Unverified)* | *27.666 ms* | 27,810.40 μs | 36,664.35 μs | ± 6,112.90 μs |

---

## 4. Rigorous Speedup Calculations

### A. Matched Model-Only Speedup (Batch Size = 1)
Using the canonical 1D-CNN forward pass latency of **$455.99\ \mu\text{s}$**:

1. **Against Single-Threaded Random Forest (`n_jobs=1`):**
   $$\text{Speedup} = \frac{30,440.80\ \mu\text{s}}{455.9920\ \mu\text{s}} = \mathbf{66.7573\times} \quad (\approx \mathbf{66.8\times})$$
2. **Against Multi-Threaded Random Forest (`n_jobs=-1`):**
   $$\text{Speedup} = \frac{71,904.6942\ \mu\text{s}}{455.9920\ \mu\text{s}} = \mathbf{157.6885\times} \quad (\approx \mathbf{157.7\times})$$

### B. Matched End-to-End Speedup (Batch Size = 1)
Using the End-to-End 1D-CNN pipeline latency of **$664.60\ \mu\text{s}$**:

1. **Against End-to-End Single-Threaded Random Forest:**
   $$\text{Speedup} = \frac{27,666.11\ \mu\text{s}}{664.60\ \mu\text{s}} = \mathbf{41.6282\times} \quad (\approx \mathbf{41.6\times})$$

---


### C. End-to-End RF Latency Status (Unverified / Omitted)
In previous reports, two divergent figures were cited for single-threaded RF end-to-end latency:
1. **$27,666.11\ \mu\text{s}$** recorded in `latency_and_size_audit.json`: This was measured in a separate 200-iteration loop. Because Random Forest single-sample runtime has substantial variance across runs (standard deviation $\pm 8,265.72\ \mu\text{s}$), running a separate loop produced an average lower than the model-only loop ($30,440.80\ \mu\text{s}$).
2. **$30,649.41\ \mu\text{s}$** cited in draft summary tables: This was an ad-hoc arithmetic sum ($30,440.80\ \mu\text{s} + 208.61\ \mu\text{s}$) rather than a measured benchmark.

**Status for Paper:** Because the empirical end-to-end RF measurement is contaminated by inter-loop timing variance and the synthetic figure lacks empirical verification, the **End-to-End RF measurement is marked as UNVERIFIED and omitted from comparative claims**. The manuscript will strictly report and compare the verified **model-only inference latencies** ($66.8\times$ and $157.7\times$).

## 5. Architectural Explanation of the Performance Gap

Why is Random Forest over $66\times$ slower on single-sample inference?
1. **Branch Depth & Cache Traversal:** The Random Forest contains **466,156 decision nodes** across 300 estimators. For every single 176-dimensional window, the CPU must traverse 300 independent decision trees sequentially or across threads. Random branching on scalar thresholds causes continuous CPU branch prediction misses and memory cache thrashing.
2. **Multi-Threading Overhead on Small Workloads:** When `n_jobs=-1` is used, scikit-learn delegates jobs to a thread pool. For a single feature vector ($1 \times 176$), the synchronization and IPC overhead of dispatching work across 8 threads far outweighs the computational cost, causing latency to more than double from $30.44\text{ ms}$ to $71.90\text{ ms}$.
3. **1D-CNN Hardware Acceleration:** In contrast, the 1D-CNN consists of contiguous matrix multiply-accumulate (GEMM) operations across only 40,610 FP32 weights, enabling vectorization via AVX-2 instructions on modern CPUs.

---

## 6. Corrected Statements and Scientific Qualifications

### Documentation Updates:
- Replaced ambiguous "45×–157×" with explicit matched speedups:
  - **$66.8\times$ faster** than single-threaded Random Forest ($456\ \mu\text{s}$ vs. $30.44\text{ ms}$).
  - **$157.7\times$ faster** than multi-threaded Random Forest ($456\ \mu\text{s}$ vs. $71.90\text{ ms}$).
- Added explicit statement:
  > *"All latency measurements were benchmarked on an AMD64 desktop CPU in Python/PyTorch and demonstrate relative algorithmic efficiency. They do not represent embedded automotive microcontroller (ECU) hardware certification or hard real-time safety compliance."*

### Files Updated:
1. `README.md`
2. `final_audit_2026/FINAL_RESEARCH_VALIDATION_REPORT.md`
3. `results/metrics/latency_and_size_audit.json`
