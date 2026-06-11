# AI Harnessing, Development Log & Transcript

This document captures the full development history of the BhuMe land-boundary correction pipeline — from initial research, to architecture decisions, to iterative failure-mode fixes. It demonstrates how the engineer directed an AI assistant to build a Gold/Platinum-tier solution.

---

## 1. Problem Understanding Research

Before writing any code, the engineer researched Maharashtra cadastral projections, Cassini drift, and georeferencing correction techniques in external AI chats:

- [ChatGPT Session 1](https://chatgpt.com/share/6a2a7dfa-ccac-8323-bbff-196a22b8d166) — Understanding the problem statement, and creating a basic code implementation logic.
- [Claude Session 2](https://claude.ai/share/e4cdace5-c83a-4789-b459-201021bb9e11) — Improving alignment performance, confidence modeling, and calibration scoring.

**Key findings from research:**

- Maharashtra cadastral boundaries drift up to 30m relative to satellite imagery due to Cassini projection scaling errors during digitization.
- Vadnerbhairav (median area ~7,750m²) has repetitive agricultural patterns but clear borders. Malatavadi (median area ~872m²) has dense small parcels and high boundary noise — they need different strategies.
- The scoring rubric explicitly rewards **restraint** (not moving already-correct plots), **calibration** (Spearman rank correlation + AUC), and **accuracy** (median IoU).

---

## 2. Chronological Prompts & AI Steering

### Prompt 1 — Requirements Analysis & Architecture Constraints

The engineer did not ask for code right away. They forced the AI into a planning phase with full constraint capture:

```text
You are working on the BhuMe land-boundary correction take-home.

Before writing any code:

1. Read and understand the entire assessment document provided in the repository.
2. Treat the assessment instructions as the primary source of truth.
3. Extract and summarize:
   * Input files and schema
   * Output schema
   * Scoring criteria
   * Expected deliverables
   * Confidence requirements
   * What differentiates Bronze, Silver, Gold, and Platinum

Goal:
Build the strongest possible solution that can realistically achieve Gold or Platinum according to the assessment rubric.

Important constraints:
* Do not optimize for toy examples.
* Do not hardcode plot IDs.
* Do not hand-edit geometries.
* Do not build a solution that only works on one village.
* Generalization is mandatory.
* Confidence calibration is a first-class objective, not an afterthought.
* Every design choice should be justified by the scoring rubric.
...
```

**Steering strategy**: Outlined phased expectations (Phase 1: pipeline I/O, Phase 2: alignment method, Phase 3: confidence, Phase 4: generalization) and required the AI to summarize inputs/scoring before touching any code.

---

### Prompt 2 — Design Approval & Phase 1 Execution

After the AI produced a research and architecture plan, the engineer approved and triggered implementation:

```text
the plan looks good. begin implementing. the code needs to clean and understandable. also the transcript  ## Problem Understanding
https://chatgpt.com/share/... https://chatgpt.com/share/6a2a7dfa-ccac-8323-bbff-196a22b8d166 add these as problem understanding as said above in the transcript folder. begin implementing
```

**Steering strategy**: Engineer linked external research sessions into the transcript directory, ensured code clarity was a hard requirement, and approved the unsupervised global shift approach.

---

### Prompt 3 — Failure Mode Diagnosis & Targeted Fixes

After running the initial pipeline on example truths and observing where it failed, the engineer manually profiled each failure, then issued a precise, structured fix directive:

```text
You are improving a cadastral boundary correction pipeline (main.py). Below are the exact failure modes
diagnosed from ground truth analysis. Apply targeted fixes that cure each failure mode without
destabilizing what already works. Do not change the overall architecture
(coarse-to-fine cross-correlation → local alignment → confidence modeling → flagging).

## DIAGNOSED FAILURE MODES

### Vadnerbhairav (Nashik)

**Failure 1: Coarse search window is too narrow**
The coarse search uses ±15m. Two truth plots (1710 and 2647) require dy shifts of 18.4m and 17.9m
respectively — both outside the search window.
Fix: Expand the coarse search window from ±15m to ±25m, keeping the step size at 3m.

**Failure 2: Global Gaussian prior centered at (0,0) suppresses large correct shifts**
The prior pulls the global estimate toward zero, causing dy to be consistently underestimated.
Fix: Re-center the fine search and local L2 prior around the coarse peak, not (0,0).

**Failure 3: Plot 622 flagged despite IoU=0.824**
Sparse boundary pixels near plot 622 yield a low norm_corr, dropping confidence below threshold.
Fix: Add a moderate-shift fallback — if local shift < 8m and norm_corr > 0.20, correct anyway.

### Malatavadi (Dharwad)

**Failure 4: Single global shift fails due to non-uniform georeferencing warp**
Cell 0 needs (7.5, 5.5), Cell 1 (-4.5, 2.0), Cell 2 (-1.5, -2.0), Cell 3 (-4.0, -1.0).
Fix: Enable 2x2 spatial grid clustering — estimate shift per cell independently.

**Failure 5: Repetitive small plots collapse confidence**
Distinctness is universally low in dense small parcels, killing confidence for all correct alignments.
Fix: Suppress distinctness penalty for small-plot villages (median area < 3000m²).

**Failure 6: Fixed area mismatch baseline is not village-adaptive**
The penalty is relative to 1.0, but Malatavadi's median ratio is 0.73 — distorting all scores.
Fix: Compute village-level median area ratio; penalize deviation from that, not from 1.0.

## CONSTRAINTS — DO NOT BREAK THESE
1. Clustering must activate on data properties, not hardcoded village names.
2. All 5 currently-correct Vadnerbhairav truth plots must remain corrected.
3. Do not change the overall confidence formula structure.
```

**Steering strategy**: Gave the AI zero guesswork by providing fully enumerated failure modes with root causes, fixes, and negative constraints. This prevented regression while driving forward progress.

---

### Prompt 4 — Calibration Regression Fix

When the fallback rule collapsed Spearman rank correlation from 0.616 to −0.26, the engineer caught it immediately and demanded a fix while also constraining execution:

```text
continue. and dont run the file yourself, give me the code and ill run it. currently Vadnerbhairav
scored 1/1 on calibration, currently it scores -0.26. we also need to fix that
```

**Steering strategy**: Took control of execution ("give me the code and ill run it") to keep the local environment stable. Pinpointed the exact metric regression and demanded targeted correction.

---

### Prompt 5 — Documenting Scorecard Results

After obtaining the final evaluation outputs from the test runs, the engineer instructed the AI to update the project documentation with the verified scores:

```text
update the recent test values into the readme files and in the transcript
```

**Steering strategy**: Ensured that the main user-facing repository README and the internal development log/transcript files accurately reflect the final performance metrics achieved on Vadnerbhairav and Malatavadi.

---

## 3. Architectural Decisions

### A. Unsupervised Coarse-to-Fine Global Shift

- **Why**: The evaluation hidden set has no `example_truths.geojson`. The pipeline must estimate georeferencing offset without labels.
- **How**: Sample 100 random plots, rasterize boundaries, grid-search over ±25m (3m step). Fine-search ±2.5m (0.5m step) around coarse peak.
- **Prior**: Gaussian prior centered on coarse peak (σ=25m coarse, σ=15m fine) penalizes outlier drift.

### B. Local Alignment with L2 Regularization

- **Why**: Agricultural field patterns cause aliasing — plot edges snap to neighboring crop row boundaries.
- **How**: Cross-correlate plot boundary raster against `boundaries.tif` in ±20m window. Gaussian prior (σ=8m) centered at global shift prevents wandering.

### C. Spatial 2×2 Grid Clustering

- **Why**: Malatavadi's georeferencing warp is non-uniform (up to 12m variation across the village).
- **How**: Triggers when `median_area < 3000m² AND plot_count > 1500`. Partition bounding box into 4 cells, estimate shift per cell from 100-plot sample.

### D. Calibrated Confidence Scoring

| Component                | Large plots                                 | Small plots             |
| ------------------------ | ------------------------------------------- | ----------------------- |
| `norm_corr` factor       | Piecewise linear (floor 0.15, ceiling 0.30) | Same                    |
| Distinctness             | Raw multiplication                          | Suppressed (high noise) |
| Displacement penalty     | Linear decay beyond 10m                     | Same                    |
| Area discrepancy penalty | Relative to village median ratio            | Same                    |

The moderate-shift fallback assigns `min(0.12, norm_corr × 0.4) × penalties` — deliberately below the 0.15 threshold scale, ensuring rank ordering is preserved for Spearman calibration.

---

## 4. Iterative Results

| Version                                            | Village       | Coverage        | Median IoU | Spearman            | AUC   |
| -------------------------------------------------- | ------------- | --------------- | ---------- | ------------------- | ----- |
| Baseline (quickstart)                              | Vadnerbhairav | 0 / 6 corrected | 0.612      | —                   | —     |
| v1: Unsupervised shift + local L2 prior            | Vadnerbhairav | 5 / 6 corrected | 0.784      | 0.616               | 1.000 |
| v2: ±25m coarse, prior re-centering, fallback rule | Vadnerbhairav | 6 / 6 corrected | 0.868      | −0.091 (regression) | —     |
| v3: Raw distinctness, fallback scaling fix         | Vadnerbhairav | 6 / 6 corrected | **0.868**  | **0.83**            | —     |
| v1: Single global shift                            | Malatavadi    | 0 / 3 corrected | 0.510      | —                   | —     |
| v2: 2×2 clustering + distinctness suppression      | Malatavadi    | 1 / 3 corrected | 0.560      | —                   | —     |
| v3: Final calibrated pipeline                      | Malatavadi    | 1 / 3 corrected | **0.783**  | —                   | —     |

---

## 5. Final Scorecard

### Vadnerbhairav

```
coverage:    6 corrected · 0 flagged · of 6 truths
accuracy:    median IoU pred=0.868 vs official=0.612  (improvement=+0.236, 100% of plots improved)
             median centroid err=5.2 m · accurate(IoU>=.5)=100%
calibration: Spearman(conf,IoU)=0.83  (rank corr, -1 to 1)
```

### Malatavadi

```
coverage:    1 corrected · 2 flagged · of 3 truths
accuracy:    median IoU pred=0.783 vs official=0.510  (improvement=+0.273, 100% of plots improved)
             median centroid err=3.4 m · accurate(IoU>=.5)=100%
calibration: — (flat confidence → no signal)
```
