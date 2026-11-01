# Prioritized Remaining Work Plan (Target: November 12, 2026)

**Project Title:** Robust Federated Learning-Based Intrusion Detection for Heterogeneous Vehicular CAN Networks  
**Audit Date:** October 31, 2026  
**Implementation Freeze Deadline:** November 8, 2026 (Leaving 4 days for final paper editing in Prism)  
**Final Submission Deadline:** November 12, 2026  
**Auditor:** Independent Senior Research Systems Auditor & Academic Reviewer  
**Output Location:** `final_audit_2026/REMAINING_WORK_PLAN.md`

---

## 1. Work Prioritization Principles

Under our strict 12-day deadline (Oct 31 – Nov 12, 2026), we must enforce ruthless scientific minimalism:
- **Do NOT** implement complex features that do not directly address a core research question or reviewer objection.
- **Do NOT** run hardware trace-replay or complex dual-variable algorithms (SCAFFOLD) that require weeks of calibration.
- **DO** eliminate factual discrepancies between documentation and code.
- **DO** complete multi-seed baselines so all paper tables display uniform statistical confidence intervals (Mean $\pm$ Std).
- **DO** implement Multi-Krum if feasible, as reviewers in FL security expect a distance-based defense baseline.

---

## 2. Priority Classification Framework

- **P0 — Critical (Must complete before paper writing begins):** Essential fixes and baseline multi-seed runs needed to freeze experimental tables.
- **P1 — Important (Should complete before Nov 8):** Significant scientific strengthening (Multi-Krum and Classical ML baseline) that directly neutralizes anticipated reviewer objections.
- **P2 — Optional (Implement only if ahead of schedule):** Non-essential defenses or attack variations (FoolsGold, Backdoors, FedProx).
- **P3 — Future Work (Explicitly out of scope):** Items deferred to future publications or journal extensions.

---

## 3. Prioritized Task Roadmap

| Priority | Task Identifier | Task Description | Why Necessary | Files Affected | Estimated Effort | Expected Deliverable / Output |
|---|---|---|---|---|:---:|---|
| **P0** | `TASK-P0-1` | **Reconcile README & Config Discrepancy** | `README.md` erroneously claims malicious clients are `[0, 1]` with $\gamma=1.5$. Code and results evaluate `[8, 9]` with $\gamma=1.0$ (39.65% data weight). | `README.md` | 1 hour | Corrected README matching exact code and experiment JSON artifacts. |
| **P0** | `TASK-P0-2` | **Multi-Seed Runs for IID & Non-IID Baselines** | Centralized, IID FedAvg, and Non-IID ($\alpha \in \{1.0, 0.5, 0.1\}$) were only evaluated on Seed 42. Seeds 43 and 44 must be executed. | `scratch/run_baselines_multiseed.py`, `results/metrics/` | 4 hours (CPU compute) | Standardized Mean $\pm$ Std across all 3 seeds for Table 1 and Table 2 in paper. |
| **P0** | `TASK-P0-3` | **Generate Per-Attack Recall Plot for Attack B** | `robust_per_attack_recall.png` only displays Label Flipping. Model-Update Corruption requires an identical companion plot. | `scratch/plot_robust_breakdown.py`, `results/figures/` | 2 hours | 2-panel figure showing per-attack recall under both Attack A and Attack B. |
| **P0** | `TASK-P0-4` | **System & Threat Architecture Diagram** | Paper Section 3 requires a publication-ready conceptual diagram of in-vehicle CAN bus, client partition, attacks, and defense. | `results/figures/architecture_diagram.pdf` | 4 hours | High-resolution vector block diagram for paper. |
| **P1** | `TASK-P1-1` | **Multi-Krum Robust Aggregation Implementation** | Reviewers expect a distance-based Byzantine defense baseline (Blanchard et al., NeurIPS 2017) to contrast with coordinate-wise Median/Trimmed Mean. | `src/federated/aggregation.py`, `tests/test_robust_aggregation.py` | 1.5 days | Multi-Krum aggregator function, unit tests, and 9-condition evaluation on seeds [42, 43, 44]. |
| **P1** | `TASK-P1-2` | **Classical ML Baseline (Random Forest)** | A skeptical reviewer will ask whether a 1D-CNN is necessary over standard tabular models on 11 CAN features. | `scratch/evaluate_classical_baselines.py`, `results/metrics/` | 1 day | Centralized Random Forest benchmark metrics (F1, Accuracy, Latency) for Table 1. |
| **P2** | `TASK-P2-1` | **FoolsGold Aggregation** | Cosine-similarity defense against synergistic Sybil attacks (Fung et al., 2018). | `src/federated/aggregation.py` | 2 days | Optional extra column in robust defense evaluation. |
| **P2** | `TASK-P2-2` | **FedProx ($\mu$-Proximal Regularization)** | Regularizes client drift under extreme non-IID skew ($\alpha=0.1$). | `src/federated/client.py` | 1.5 days | Optional comparison against standard local SGD under extreme skew. |
| **P2** | `TASK-P2-3` | **Targeted CAN Backdoor Injection** | Injects a specific CAN ID trigger (`0x02A0`) into benign windows. | `src/federated/threat.py` | 2 days | Demonstrates stealthy backdoor resilience. |
| **P3** | `TASK-P3-1` | **SCAFFOLD Control Variates** | Dual-variable variance reduction for FL drift. | `src/federated/` | 4 days | **DEFERRED:** Doubles client-server communication footprint. |
| **P3** | `TASK-P3-2` | **Hardware Trace-Replay Testbed** | Embedded execution on physical Raspberry Pi 4 with MCP2515 CAN SPI transceiver. | Hardware Lab | 2 weeks | **DEFERRED:** Dedicated hardware follow-up paper. |
| **P3** | `TASK-P3-3` | **Cross-Vehicle Multi-Dataset Benchmark** | Evaluating ROAD or Survival datasets alongside HCRL. | Preprocessing | 1 week | **DEFERRED:** Out of current Kia Soul paper scope. |

---

## 4. Proposed Timeline Leading to November 12, 2026

```text
[Oct 31 - Nov 1]  P0 Fixes: README correction, Multi-seed baseline runs (Seeds 43, 44), companion figures.
[Nov 2 - Nov 3]   P1 Multi-Krum Aggregator: Implementation, unit testing, multi-seed evaluation.
[Nov 4]           P1 Classical Baseline: Random Forest evaluation on centralized CAN dataset.
[Nov 5]           IMPLEMENTATION FREEZE: All code, metrics JSON, and figures locked.
[Nov 6 - Nov 9]   Paper Drafting in Prism: Sections 1-6 (Intro, Related Work, Threat Model, System, Results).
[Nov 10 - Nov 11] Internal Peer-Review & Skeptical Reviewer Rebuttal Polish (Sections 7-8).
[Nov 12]          Final Camera-Ready Compilation & Conference Submission.
```

---

## 5. Decision on Roadmap Cuts

To guarantee a defensible, high-impact paper by November 12, the following items are **officially cut** from the current submission scope:
1. **SCAFFOLD:** Implementing client and server state control variates introduces significant complexity without altering the primary thesis on Byzantine robust aggregation.
2. **Hardware Trace Replay:** Profiling on real-time CPU ($455.99\ \mu\text{s}$) already satisfies the sub-millisecond CAN deadline. Physical hardware demonstration can be listed as Future Work.
3. **Adaptive Backdoor Attacks:** Label flipping and delta sign reversal already demonstrate the complete spectrum from data poisoning to model poisoning with total FedAvg failure.

