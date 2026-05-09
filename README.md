# Richter's Predictor — Earthquake Damage Classification

Predicting structural damage from the 2015 Gorkha earthquake in Nepal using
survey data from 260,601 buildings. Built as a course project and submitted to
the [DrivenData competition](https://www.drivendata.org/competitions/57/nepal-earthquake/),
achieving a **public leaderboard micro-F1 of 0.7460 (top 28%)**.

---

## Problem

Following the April 2015 earthquake, survey teams recorded building attributes
across Nepal's affected districts. The task is a three-class ordinal classification:
predict whether each building sustained **low (Grade 1)**, **medium (Grade 2)**,
or **near-complete destruction (Grade 3)**. The official evaluation metric is
micro-averaged F1 score.

Accurate predictions help disaster-response teams prioritise reconstruction
resources and inform future building codes in seismically active regions.

---

## Results

| Model | Val micro-F1 | Val macro-F1 | Grade 1 F1 | Grade 2 F1 | Grade 3 F1 |
|---|---|---|---|---|---|
| Majority-class baseline | 0.5689 | — | 0.00 | — | 0.00 |
| Decision Tree (depth=5) | ~0.68 | — | — | — | — |
| Logistic Regression — FS1 | 0.6598 | 0.5386 | 0.329 | 0.745 | 0.542 |
| Random Forest (n=100) — FS1 | 0.7472 | 0.6949 | 0.596 | 0.795 | 0.694 |
| Random Forest (n=100) — FS2 | 0.7481 | 0.6963 | 0.597 | 0.795 | 0.697 |
| CatBoost (tuned) — FS2 | 0.7467 | 0.6955 | 0.598 | 0.794 | 0.694 |
| **Random Forest (tuned) — FS2** | **0.7501** | **0.6963** | **0.594** | **0.798** | **0.698** |

**DrivenData public leaderboard: 0.7460 — top 28%**

---

## Approach

The project follows a five-phase structure documented end-to-end in a single
Jupyter notebook. Every modelling decision is justified by a visual or metric
produced in the previous step.

### Phase 1 — Exploratory Data Analysis
- Target distribution revealed class imbalance (Grade 2 = 57%, Grade 1 = 10%),
  motivating micro-averaged F1 as the primary metric over accuracy
- Stacked bar charts per categorical feature identified `foundation_type` and
  `roof_type` as the strongest construction-level predictors
- A damage-delta analysis across all superstructure material flags showed
  RC-engineered construction is the most protective material; mud-mortar-stone
  the most vulnerable
- District-level `groupby` aggregations revealed substantial geographic variation
  in mean damage, directly motivating the Phase 3 geo encoding

### Phase 2 — Preprocessing and Baseline
- Stratified 80/20 train/validation split preserving class proportions
- `sklearn` Pipeline with a custom `AgeCapper` transformer (caps the implausible
  outlier tail at 150 years, identified in EDA) and `OrdinalEncoder` for the
  eight string categorical columns
- Majority-class dummy classifier establishes the naive floor (micro-F1 = 0.5689)
- Depth-5 Decision Tree baseline confirms non-linearity in the problem and
  provides a first feature importance ranking consistent with EDA findings
- Standardised `evaluate_model()` function computes micro-F1, macro-F1, per-class
  F1, and a normalised confusion matrix for every model — results accumulate in a
  shared `results` dict for Phase 4's comparison table

### Phase 3 — Feature Engineering
Two feature sets were produced and carried forward:

**FS1** — geo mean-target encoding + secondary-use flag removal  
**FS2** — FS1 + superstructure cluster label + PCA components

| Engineering step | Justification | Technique |
|---|---|---|
| Geo mean-target encoding | District-level damage std = 0.24–0.40 across the three geo levels (§7) | 5-fold cross-fold encoding with smoothing to prevent leakage; unseen regions fall back to global mean |
| Secondary-use flag removal | All 10 flags had \|corr\| < 0.10 with the target (§8) | Threshold filter |
| Superstructure clustering | Silhouette jumped at k=5; damage spread across clusters = 0.71 grade points (§19) | MiniBatchKMeans on the 11 binary material flags |
| PCA on material flags | 2 components explain 56% of flag variance; PC1 is the traditional-vs-modern construction axis (§20) | sklearn PCA, 2 components retained |

A controlled Decision Tree experiment (same depth, same hyperparameters, only
features varied) showed FS1 and FS2 both outperformed the raw baseline, with FS2
providing a further marginal gain that Random Forest could exploit through deeper
interaction modelling.

### Phase 4 — Model Comparison and Tuning
- **Logistic Regression** on FS1 confirmed non-linearity: the micro-F1 gap vs
  Random Forest was >0.08, justifying the move to ensemble methods
- **Random Forest** on both feature sets with `RandomizedSearchCV`
  (n_iter=10, 5-fold CV): best params `max_features='log2'`,
  `min_samples_leaf=2`, `max_depth=30`
- **CatBoost** introduced after RF tuning plateaued — gradient boosting's
  sequential residual correction targets the hard cases (Grade 1) that
  averaging-based methods handle less precisely; tuned with
  `RandomizedSearchCV` and final re-train using early stopping
  (best: `depth=8`, `learning_rate=0.1`, `l2_leaf_reg=5`, 298 trees)
- Final comparison table generated from live `results` dict — no copy-pasted
  numbers

### Phase 5 — Interpretability with SHAP
- `shap.TreeExplainer` with `feature_perturbation='tree_path_dependent'`
  on a stratified 500-building sample from the validation set
- SHAP values cached to disk after first computation — reloads in < 1 second
  after any kernel restart
- **Global:** mean absolute SHAP bar chart compared side-by-side with Gini
  importance — rankings are consistent across both methods, validating the
  feature engineering decisions
- **Directional:** beeswarm plot across all three damage grade classes —
  high geo-risk encoding strongly pushes toward Grade 3; RC-engineered
  construction pushes away from it
- **Local:** waterfall plots for a confident correct Grade 3, a confident
  correct Grade 1, and a representative misclassification — shows concretely
  where the model's blind spots are

---

## Key Findings

**What drives damage prediction:**

1. **Geographic location** is the dominant signal. The three geo-encoded features
   (`geo_level_3`, `geo_level_2`, `geo_level_1` mean damage encoding) account
   for roughly 52% of combined Gini importance and top the SHAP ranking. Location
   is a proxy for proximity to the epicentre, local geology, and the prevailing
   building tradition in each area.

2. **Building age** is the most important building-level feature. The SHAP
   dependence plot shows a monotone relationship: older buildings sustained more
   severe damage, consistent with construction predating seismic codes.

3. **Footprint area** is negatively associated with Grade 3 damage — smaller
   buildings were more vulnerable, likely due to less structural redundancy.

4. **Superstructure material** adds meaningful signal once geographic variation
   is controlled for. RC-engineered construction is strongly protective; mud-mortar-
   stone is the most vulnerable material and accounts for 76% of the dataset.

**The persistent challenge — Grade 1 F1 ≈ 0.59:** Buildings that survived
relatively intact in heavily damaged regions are the hardest to identify.
The model over-weights geographic risk and under-weights protective construction
features in the worst-affected areas. This is the most operationally significant
limitation: a near-destroyed building misclassified as low-damage is a
dangerous error in a real deployment.

---

## Repository Structure

```
modeling-earthquake-damages/
├── earthquake_damage.ipynb   # Full end-to-end notebook (phases 1–5)
├── data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   └── test_values.csv
├── images/
│   ├── phase1/               # EDA figures
│   ├── phase2/               # Baseline evaluation plots
│   ├── phase3/               # Feature engineering diagnostics
│   ├── phase4/               # Model comparison and tuning results
│   └── phase5/               # SHAP global, beeswarm, dependence, local plots
├── shap_cache/
│   └── shap_values.npy       # Cached SHAP output (avoids 21-min recompute)
└── README.md
```

---

## Reproducing the Results

**Requirements**
```
python >= 3.9
scikit-learn
pandas
numpy
matplotlib
seaborn
catboost
shap >= 0.41
```

Install dependencies:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn catboost shap
```

**Run the notebook**

Place the three competition CSV files in `data/` and run
`earthquake_damage.ipynb` top to bottom. The SHAP cell will compute values on
first run (~20 minutes for a 200-tree Random Forest on a laptop CPU) and cache
them to `shap_cache/shap_values.npy` — subsequent runs load the cache instantly.

All random states are fixed at `seed=42` throughout. The stratified split,
cross-fold geo encoding, and `RandomizedSearchCV` are all deterministic given
the same seed, so scores should reproduce exactly.

---

## Competition

**Platform:** [DrivenData — Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/)  
**Metric:** Micro-averaged F1 score  
**Public leaderboard score:** 0.7460  
**Leaderboard rank:** Top 28%  
**Leaderboard ceiling:** 0.7558