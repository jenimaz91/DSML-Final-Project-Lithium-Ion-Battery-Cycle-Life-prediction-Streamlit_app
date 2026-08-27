# Predicting Li-ion Battery Cycle Life from the First 100 Cycles

Predicting how long a lithium-ion cell will last using only data from its first 100 cycles — before any meaningful capacity loss is visible.

Reproduction and extension of Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, **Nature Energy** 4, 383–391 (2019).
StreamLit Live app: http://localhost:8501/
---
**(Summary: 
 A linear model predicts battery cycle life from the first 100 cycles at 12.7% MAPE on a production run it has never seen. Gradient Boosting scored higher in cross-validation and then failed completely on that same unseen data (r² −1.389), because tree ensembles cannot extrapolate beyond their training range. The train–test gap predicted this; the cross-validated mean did not. Full results below.)**

## The problem

Qualifying a new cell design or charging protocol means cycling it to failure. For the cells in this dataset that is between 148 and 2,300 cycles — weeks to months of occupancy on expensive test equipment, per cell, per protocol.

The commercial consequence is that manufacturers can only evaluate a small number of designs, because each evaluation is measured in machine-months. A model that predicts end of life from the first 100 cycles compresses that loop by roughly an order of magnitude and turns cycle-life testing from a bottleneck into a screening step.

The difficulty is that at cycle 100 there is almost nothing to see. Capacity fade is still within measurement noise, and a cell that will die at 636 cycles is visually indistinguishable from one that will reach 1,074. The signal must come from somewhere other than the capacity trend.

## The dataset

124 commercial A123 APR18650M1A cells (LFP/graphite, 1.1 Ah nominal), cycled to failure under 72 fast-charging protocols with an identical 4C discharge. Published by MIT, Stanford and the Toyota Research Institute; available at [data.matr.io](https://data.matr.io/1/).

| Batch | File date | Raw records | After cleaning |
|---|---|---|---|
| 1 | 2017-05-12 | 46 | 36 |
| 2 | 2017-06-30 | 48 | 43 |
| 3 | 2018-04-12 | 46 | 43 |
| | | **140** | **122** |

End of life is the cycle at which discharge capacity falls to 80% of nominal — 0.88 Ah.

The files are MATLAB v7.3, which is HDF5 underneath. `scipy.io.loadmat` cannot read them; `h5py` is required, and every field is stored as an HDF5 object reference that must be dereferenced through the file handle (`f[batch['summary'][i, 0]]`, materialised with `[()]` — `.value` was removed in h5py 3.x).

Each cell provides per-cycle summary statistics and, critically, `Qdlin`: discharge capacity interpolated onto a fixed 1,000-point voltage grid from 3.5 V to 2.0 V. Because that grid is identical for every cycle and every cell, capacity curves from different cycles can be subtracted pointwise.

---

## Workflow

| Part | Stage | Output |
|---|---|---|
| 1 | Data structure orientation | HDF5 layout, one cycle inspected, `Qdlin` grid understood |
| 2 | Cell inventory + data quality audit | 140 records → 123 after label-quality exclusions |
| 3 | Feature engineering | 5 candidate features computed; `b3c37` removed → **122 cells** |
| 4 | **Experiment 1** — train batches 1–2, test sealed batch 3 | Generalisation to an unseen production run |
| 5 | **Experiment 2** — all batches pooled, 5-fold CV | Signal content when populations match |
| 6 | Model selection | Linear regression, 8 features |
| 7 | Uncertainty quantification | ±27% interval, coverage measured on batch 3 |
| 8 | Artefact export | 5 files, ~1 MB, consumed by the Streamlit app |

Two ordering decisions are deliberate:

**Features are computed before cleaning is complete.** Two exclusion criteria depend on feature values — `b3c37` is caught by a z-score on `log_var_dQ`, and the stitched-cell diagnosis uses capacity relationships. Cleaning first would split feature extraction across two parts of the notebook.

**The evaluation protocol is declared before any model is fitted.** The sealed split, the metrics, and the commitment to open batch 3 exactly once are all fixed in advance. This is what makes the covariate-shift result in Part 4 a finding rather than a rationalisation.

---

## Results

### Experiment 1 — train on batches 1–2, test on sealed batch 3

Batch 3 was manufactured and cycled roughly a year after batches 1–2 and was untouched until final evaluation. This simulates deploying the model on a **future production run**.

Development set 79 cells; sealed secondary set 43 cells.

**Cross-validated on the development set (5-fold):**

| Model | Features | Train r² | CV r² | CV std | Gap |
|---|---|---|---|---|---|
| GradBoost | 8 | 0.999 | **0.810** | 0.051 | 0.189 |
| RandomForest | 8 | 0.967 | 0.801 | 0.082 | 0.166 |
| RandomForest | 2 | 0.970 | 0.796 | 0.062 | 0.174 |
| GradBoost | 2 | 0.997 | 0.794 | 0.042 | 0.203 |
| Linear | 8 | 0.868 | 0.783 | 0.063 | **0.085** |
| Ridge | 8 | 0.867 | 0.782 | 0.057 | 0.085 |
| Ridge | 2 | 0.810 | 0.748 | 0.082 | 0.062 |
| Linear | 2 | 0.810 | 0.746 | 0.083 | 0.064 |
| Linear | 1 | 0.748 | 0.660 | 0.121 | 0.088 |

Gradient Boosting leads. A `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate` and `min_samples_leaf` returned a best CV r² of 0.810 — identical to the untuned model, so the ranking is not a tuning artefact.

**Sealed evaluation on batch 3:**

| Model | Features | Batch-3 r² | Batch-3 MAPE |
|---|---|---|---|
| Ridge | 1 | **+0.590** | 12.97% |
| Linear | 1 | +0.585 | 13.24% |
| **Linear** | **8** | **+0.557** | **12.74%** |
| Linear | 2 | +0.499 | 12.91% |
| Ridge | 2 | +0.483 | 13.15% |
| Ridge | 8 | +0.458 | 14.17% |
| GradBoost | 1 | +0.259 | 15.37% |
| RandomForest | 1 | +0.080 | 17.04% |
| GradBoost | 2 | −0.319 | 19.41% |
| RandomForest | 2 | −0.426 | 19.28% |
| RandomForest | 8 | −0.978 | 23.79% |
| GradBoost | 8 | **−1.389** | 26.92% |

Published benchmarks on the same secondary set: 11.4% (variance model), 10.7% (full model), 8.6% (discharge model).

**The ranking inverts completely.** Gradient Boosting with 8 features ranked first in development cross-validation and last on batch 3. A negative r² means the model performs worse than predicting the training mean — the trees did not degrade, they inverted.

Adding features makes the trees monotonically worse: RandomForest goes 0.080 → −0.426 → −0.978 as features increase from 1 to 2 to 8. Every additional dimension is another axis along which batch 3 lies outside the training range.

### Why: covariate shift

**The train–test gap was the reliable warning and the cross-validated mean was not.** Every tree configuration gapped 0.166–0.303; every linear and Ridge configuration gapped 0.062–0.088. That separation predicted the outcome where mean score did not.

### Experiment 2 — all batches pooled, 5-fold cross-validation

Different question: how much signal do these features carry when training and deployment populations match? Also uses 121–122 cells for training rather than 79.

| Model | Features | n | Train r² | CV r² | CV std | Gap |
|---|---|---|---|---|---|---|
| Ridge | 8 | 121 | 0.894 | **0.861** | **0.016** | 0.033 |
| **Linear** | **8** | **121** | **0.894** | **0.859** | **0.017** | **0.035** |
| RandomForest | 8 | 121 | 0.978 | 0.841 | 0.030 | 0.137 |
| Ridge | 2 | 122 | 0.858 | 0.839 | 0.037 | 0.019 |
| Linear | 2 | 122 | 0.859 | 0.838 | 0.040 | 0.021 |
| RandomForest | 2 | 122 | 0.978 | 0.816 | 0.058 | 0.162 |
| Ridge | 1 | 122 | 0.830 | 0.810 | 0.025 | 0.020 |
| Linear | 1 | 122 | 0.830 | 0.810 | 0.027 | 0.020 |
| GradBoost | 8 | 121 | 0.996 | 0.807 | 0.051 | 0.189 |
| GradBoost | 2 | 122 | 0.992 | 0.797 | 0.112 | 0.195 |
| RandomForest | 1 | 122 | 0.974 | 0.795 | 0.037 | 0.179 |
| GradBoost | 1 | 122 | 0.980 | 0.775 | 0.072 | 0.205 |

**These scores are not an improvement on Experiment 1.** Pooling does not fix covariate shift; it removes the shift from the *evaluation* by placing batch-3 cells on both sides of every split. Experiment 1 estimates performance on an unseen production run; Experiment 2 estimates feature quality.

### Feature decomposition

| Feature set | n | CV r² |
|---|---|---|
| `log_var_dQ` | 122 | 0.810 ± 0.027 |
| `Q_initial` | 122 | **−0.013** ± 0.029 |
| policy (6 features) | 121 | **0.240** ± 0.190 |
| `log_var_dQ` + `Q_initial` | 122 | 0.838 ± 0.040 |
| `log_var_dQ` + policy | 121 | 0.828 ± 0.014 |
| all 8 | 121 | **0.859** ± 0.017 |

`Q_initial` alone predicts worse than the mean. The six charging-protocol parameters alone reach only 0.240 with a standard deviation nearly as large. Together with `log_var_dQ` they add 0.049. **The components are individually near-worthless and jointly useful** — which licenses using them and forbids interpreting their coefficients.

### Neural network

A feed-forward network (16 → 8 → 1, ReLU, Adam 0.01, early stopping) on two features, 68 training cells / 17 validation / 37 test.

| | Test r² | MAPE |
|---|---|---|
| NN seed 0 | 0.733 | 14.94% |
| NN seed 7 | 0.738 | 13.91% |
| NN seed 42 | 0.765 | 12.75% |
| **Linear (2 feat, same split)** | **0.822** | **11.81%** |

The linear model beats all three networks. The spread across seeds — identical data, identical architecture, differing only in random weight initialisation — is 0.032, so the network has no single score to report. Train r² 0.921 against test 0.706 in the primary run.

An 8-feature network was attempted and failed to converge (r² of −0.7 to −8.1 at learning rate 0.01); it is not reported.

The result is expected at n = 122: 217 parameters against 68 training cells, on a relationship that is close to linear (r = −0.911) once both axes are logged.

---

## Model selection

**Linear regression on 8 features:** `log_var_dQ`, `Q_initial`, `c1`, `soc_switch`, `c2`, `c_avg`, `c_max`, `charge_t`.

| Criterion | Evidence |
|---|---|
| Highest cross-validated r² among stable models | 0.859 pooled |
| Tightest fold-to-fold variance | ± 0.017, against 0.030–0.112 for tree models |
| Smallest gap among competitive models | 0.035, against 0.137–0.205 |
| **Only family that survives the sealed evaluation** | **+0.557 against −0.978 and −1.389** |

The fourth criterion is decisive. Ridge scores 0.861 to Linear's 0.859 — a difference of 0.002 against fold spreads of 0.016, which is noise — but Ridge's batch-3 score is 0.458 against Linear's 0.557. Unregularised linear regression is selected.

### Coefficients, and how far they can be read

| Feature | Standardised coefficient |
|---|---|
| `log_var_dQ` | **−0.187** |
| `Q_initial` | +0.034 |
| `c_max` | +0.033 |
| `soc_switch` | +0.026 |
| `c1` | −0.019 |
| `c_avg` | +0.018 |
| `c2` | +0.015 |
| `charge_t` | +0.012 |

`log_var_dQ` dominates at 5.5× the next largest, with the correct negative sign: more early-cycle voltage-curve distortion, shorter life. `Q_initial` is positive and physically sensible.

**The charging-parameter signs are physically wrong.** `c_max` is positive, implying harder charging predicts longer life. It does not. The magnitudes are near zero, the parameters are intercorrelated by construction (`c_avg`, `c_max` and `charge_t` are all functions of `c1`, `c2` and `soc_switch`), and the decomposition above shows they reach only 0.240 alone.

**This model should not be described as physically interpretable beyond `log_var_dQ` and `Q_initial`.** The protocol features earn their place empirically and on that basis only.

---

## Feature engineering

### The primary feature: `log_var_dQ`

Rather than tracking capacity, subtract two full discharge curves:

```
ΔQ(V) = Q₁₀₀(V) − Q₁₀(V)
```

across all 1,000 grid points, then take `log₁₀(var(ΔQ))`.

This discards the capacity axis and interrogates the **voltage** axis. Total capacity at cycle 100 is essentially unchanged, but *where in the voltage window* it is delivered has shifted, and the shift concentrates in a narrow band. A typical curve is flat from 3.5 to 3.2 V, drops sharply through 3.2–3.1 V, then partially recovers. That region is where the LFP two-phase plateau meets the graphite staging transitions — the part of the discharge most sensitive to loss of cyclable lithium.

**r = −0.905, r² = 0.818** against log cycle life across 123 cells (before `b3c37` removal).

### Why the target is logged

Against raw `cycle_life` the correlation is r = −0.867 (r² 0.751); against `log_life`, r = −0.905 (r² 0.818). The raw scatter is visibly curved; the logged one is straight.

The variance argument is weaker and is reported as such. Binning `log_var_dQ` into quartiles:

| bin | `cycle_life` mean | `cycle_life` CV | `log_life` std |
|---|---|---|---|
| (−5.10, −4.51] | 1524 | 0.158 | 0.069 |
| (−4.51, −3.91] | 1018 | 0.245 | 0.092 |
| (−3.91, −3.32] | 587 | 0.235 | 0.096 |
| (−3.32, −2.73] | 373 | 0.372 | 0.203 |

Both rise across the range. **Logging does not stabilise the residual variance here** — the case for it is linearity. Residual variance still grows with the feature, which is why the interval analysis in Part 7 matters.

### Candidates rejected, each on evidence

| Feature | Evidence |
|---|---|
| `min_dQ` | Correlated with `log_var_dQ`; test r² 0.674 → 0.677, no contribution |
| `IR_initial` | Test r² 0.674 → **0.525**, gap 0.090 → 0.317 |
| `ir_delta` | **53 of 122 cells negative** (43%) — resistance apparently falling. Mean −0.000033, std 0.000411, 12× larger. Test r² → **0.065**, gap **0.720** |
| `q_slope` | Linear fit explains a **median of 15%** of variation (25th percentile 2.6%); **19 of 46 cells positive** |
| `dq_skew`, `dq_kurt` | Test r² 0.674 → 0.674 and 0.676 |

The `q_slope` result deserves emphasis: **the most obvious feature — how fast capacity is falling — barely exists at cycle 100.** Its own fit is poor and its sign is wrong for 41% of cells, because what it captures is the formation/break-in rise rather than degradation. Everything must come from curve *shape* instead.

`ir_delta` fails a physical consistency check before it ever meets the target: a feature that reports the wrong direction for 43% of observations cannot be measuring what it claims to.

---

## Data quality audit

Three defects in the raw 140 records, each identified from the data and only then cross-checked against the paper's published exclusion list.

**Ten records, five physical cells — histories split across two files.** Five batch-1 cells were still under test when that file was written; their remaining life is recorded in batch 2 under different IDs. The batch-2 halves are worse than truncated: their "cycle 10" and "cycle 100" come from a cell already past 1,000 cycles, so any early-cycle feature describes an aged cell while the label describes a short life.

Nearest-capacity matching identified the correct *set* but assigned pairings unreliably — one batch-2 cell was nearest for two different batch-1 cells, and only one of five pairings agreed with the published list. The set was confirmed against three further signals, with the strongest non-listed candidate (`b2c47`) carried as a control: depressed initial capacity against batch 2's median of 1.0714 Ah, matching charging policy, and the `cycle_life = n_cycles + 1` signature. The control failed and was retained.

**Seven records — cells that never reached end of life.** Identified by `Q_last > 0.885 Ah`: a cell that genuinely crossed 0.88 Ah cannot have finished above it. Two batch-3 cells have no `cycle_life` at all despite 2,189 and 2,237 recorded cycles.

**One record — measurement noise (`b3c37`).** Flagged at z-score −3.22 on `log_var_dQ`, then confirmed visually. Its ΔQ(V) curve spikes to +0.013 Ah at 3.2 V — the cell apparently delivering over 1% *more* capacity at cycle 100 than at cycle 10 — then oscillates between +0.013 and −0.008 across a narrow voltage window. Cells do not gain capacity. It survived 1,390 cycles while recording the lowest ΔQ variance in the dataset, which would have been a maximally misleading training point.

**Result: 122 clean cells**, not the paper's 124. The difference is deliberate: every exclusion here is defensible from the data itself.

Five physical range checks all pass: `Q_initial` within 1.00–1.15 Ah, `IR_initial` within 0.010–0.025 Ω, `Q_last` ≤ 0.885 Ah, `cycle_life` > 100, and `cycle_life ≤ n_cycles + 1`.

---

## Uncertainty quantification

Residual standard deviation on the development set is **0.0532** in log-life for the selected model, giving a 95% interval of **±27%** in cycles. The two-feature model gave 0.0622, or ±32%.

| Interval | Width | Batch-3 coverage |
|---|---|---|
| ±1.96σ | ±27% | **84%** |
| ±2.5σ | ±36% | 93% |
| ±3.0σ | ±44% | 98% |

**The two-feature model produced identical coverage — 84 / 93 / 98% — with a wider interval.** A more precise model bracketing the same fraction of cells rules out imprecision as the cause.

The seven cells outside the interval confirm it:

| cell | actual | predicted | error | `log_var_dQ` |
|---|---|---|---|---|
| b3c3 | 1115 | 831 | −25% | −4.117 |
| b3c6 | 667 | 507 | −24% | −3.526 |
| b3c7 | 1836 | 1336 | −27% | −4.489 |
| b3c9 | 1039 | 782 | −25% | −3.982 |
| b3c38 | 1935 | 1139 | −41% | −4.342 |
| b3c42 | 1642 | 1150 | −30% | −4.346 |
| b3c45 | 1801 | 1403 | −22% | −4.521 |

**Every one is under-predicted.** Not a single over-prediction. If these were random errors, roughly half would fall on each side.

One-directional failure is the signature of extrapolation, not imprecision. These cells sit at the long-lived, low-variance end of the range, where the model has no training data. Better features shrink the errors of cells inside the range; only training data covering the range would fix cells outside it.

This is implemented directly in the application as an out-of-range check that withdraws the interval when a cell falls outside the training distribution.

---

## The application

```bash
pip install -r requirements.txt
streamlit run app.py
```

For each cell:

- capacity fade curve with end of life and the cycle-100 prediction point marked — the visual demonstration that nothing has yet happened when the prediction is made
- the cell's ΔQ(V) curve and its variance
- predicted versus actual cycle life with percentage error
- a 95% prediction interval
- **an out-of-range check that withdraws the interval when the cell falls outside the training distribution**

The last item is the point. Most deployed models return a number regardless of whether they have grounds for it; this one states when the number should not be trusted.

---

## Repository structure

```
.
├── project_notebook_final.ipynb              full analysis, Parts 1-8
├── app.py                       Streamlit application
├── requirements.txt             app runtime dependencies
├── README.md
└── models/                      artefacts exported by the notebook
    ├── LR_8feat_selected.pkl    selected model (n=121)
    ├── LR_2feat.pkl
    ├── GB_8feat.pkl
    ├── RF_8feat.pkl
    ├── app_data.parquet         121 cells x 23 columns
    ├── dq_curves.npz            ΔQ(V) per cell        (0.47 MB)
    ├── fade_curves.npz          capacity fade per cell (0.32 MB)
    └── vdlin.npy                shared 1,000-point voltage grid
```

Each `.pkl` stores a dict with `pipeline`, `features`, `target`, `n_train` and `sklearn_version`, so the feature order used at fit time travels with the model.

The raw `.mat` files total roughly 8 GB and are not committed.

---

## Limitations

- **One chemistry, one discharge protocol.** All cells are LFP/graphite discharged at 4C. Nothing here demonstrates transfer to NMC, other form factors, or variable discharge.
- **122 cells is small.** Fold-to-fold variance matters more than the mean; single-split results should be treated as noise. RandomForest scoring 0.787 on one pooled split and −0.978 on batch 3 is the cautionary example.
- **Batch 3 has now been used.** Further model selection against it would no longer be an honest held-out evaluation.
- **The out-of-range check is a bounding box**, testing each feature independently. A cell can fall inside all eight ranges while sitting in an unpopulated corner of the joint feature space. Leverage would catch that; the box will not.
- **The deployed model is fitted on all 121 cells**, so errors shown in the app for dataset cells are optimistic and are not held-out performance.
- **One cell is lost** to an unparsed policy string, reducing the 8-feature models to 121.
- **The neural network was not carried to the sealed evaluation.** Its behaviour under distribution shift is untested.

---

## Reproduction

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements-notebook.txt
```

Download the three batch files from [data.matr.io](https://data.matr.io/1/) into the project root, then run `project_2.ipynb` top to bottom. The final cells regenerate everything in `models/` and verify that the artefacts are mutually consistent.

## Reference

Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* **4**, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8

Final project — Ironhack Berlin, Data Science & Machine Learning bootcamp.
