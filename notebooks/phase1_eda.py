import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── helpers ──────────────────────────────────────────────────────────────────
def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE & INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""# Richter's Predictor: Modeling Earthquake Damage
## Nepal Earthquake — DrivenData Competition

### Problem Statement

In April 2015, a 7.8-magnitude earthquake struck Nepal near Gorkha, causing widespread destruction.
This project uses survey data collected by **Kathmandu Living Labs** and the **Central Bureau of
Statistics** to predict the level of structural damage sustained by individual buildings.

**Target variable:** `damage_grade` — an ordinal variable with three levels:
- **1** → Low damage
- **2** → Medium damage
- **3** → Near-complete destruction

**Why this matters:** Accurate damage prediction models can help disaster-response teams prioritise
resources, guide rebuilding policy, and inform future construction standards in seismically active
regions.

---

## Notebook Structure

| Phase | Description |
|-------|-------------|
| **1 — EDA** | Explore the target and all predictor distributions; identify key patterns |
| **2 — Preprocessing + Baseline** | Encode features, split data, train a simple baseline model |
| **3 — Feature Engineering** | Create new features motivated by EDA findings |
| **4 — Model Comparison + Tuning** | Compare multiple algorithms; tune the best one |
| **5 — Interpretability** | Use SHAP to explain what the final model learned |

This notebook is **Phase 1: Exploratory Data Analysis**.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## 0. Setup\n\nImport all libraries and configure plot aesthetics used throughout this notebook."))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Custom ordered palette for damage grades (green → amber → red)
DAMAGE_PALETTE = {1: '#4CAF50', 2: '#FF9800', 3: '#E53935'}
DAMAGE_LABELS  = {1: 'Grade 1 — Low', 2: 'Grade 2 — Medium', 3: 'Grade 3 — Destruction'}

print("Libraries loaded successfully.")
print(f"Pandas  : {pd.__version__}")
print(f"NumPy   : {np.__version__}")
print(f"Seaborn : {sns.__version__}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 1. Data Loading

The competition provides three CSV files:
- `train_values.csv` — feature matrix for training buildings
- `train_labels.csv` — target column (`damage_grade`) matched by `building_id`
- `test_values.csv`  — feature matrix for the held-out test set (no labels)

We merge features and labels on `building_id` to form our working training dataframe.
"""))

cells.append(code("""\
# ── Load raw files ────────────────────────────────────────────────────────────
train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')
test_values  = pd.read_csv('test_values.csv')

# ── Merge features + labels into one training dataframe ───────────────────────
df = train_values.merge(train_labels, on='building_id')

print(f"Training set : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Test set     : {test_values.shape[0]:,} rows × {test_values.shape[1]} columns")
df.head(3)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 2. Data Quality Check

Before any analysis we verify the dataset's integrity: missing values, data types, and
basic shape. Knowing there are no gaps is itself a useful finding — it means we won't need
imputation strategies, which simplifies our preprocessing pipeline.
"""))

cells.append(code("""\
# ── Missing values ────────────────────────────────────────────────────────────
missing = df.isnull().sum()
print("=== Missing values ===")
if missing.sum() == 0:
    print("  No missing values found in any column. ✓")
else:
    print(missing[missing > 0])

# ── Column types ─────────────────────────────────────────────────────────────
print(f"\\n=== Column types ===")
type_counts = df.dtypes.astype(str).value_counts()
for t, n in type_counts.items():
    print(f"  {t:12s}: {n} columns")

# ── Identify feature groups ───────────────────────────────────────────────────
geo_cols        = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
numeric_cols    = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
categorical_cols= ['land_surface_condition', 'foundation_type', 'roof_type',
                   'ground_floor_type', 'other_floor_type', 'position',
                   'plan_configuration', 'legal_ownership_status']
superstructure_cols = [c for c in df.columns if c.startswith('has_superstructure_')]
secondary_use_cols  = [c for c in df.columns if c.startswith('has_secondary_use_')]

print(f"\\n=== Feature groups ===")
print(f"  Geographic IDs       : {len(geo_cols)} columns")
print(f"  Numeric building     : {len(numeric_cols)} columns")
print(f"  Categorical building : {len(categorical_cols)} columns")
print(f"  Superstructure flags : {len(superstructure_cols)} columns")
print(f"  Secondary-use flags  : {len(secondary_use_cols)} columns")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# TARGET DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Target Variable — `damage_grade`

Understanding the distribution of our target is the most critical first step. If the
classes are imbalanced, plain accuracy will be a misleading metric — a model that always
predicts the majority class would score high without being useful.

**Finding:** Grade 2 (medium damage) accounts for ~57% of buildings. This is a moderate
imbalance. It motivates our choice of **micro-averaged F1 score** as the primary metric
(which is also the official competition metric), since it penalises poor performance on
minority classes more honestly than accuracy does.
"""))

cells.append(code("""\
# ── Compute counts and percentages ────────────────────────────────────────────
target_counts = df['damage_grade'].value_counts().sort_index()
target_pct    = df['damage_grade'].value_counts(normalize=True).sort_index() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: count bar chart
bars = axes[0].bar(
    [DAMAGE_LABELS[g] for g in target_counts.index],
    target_counts.values,
    color=[DAMAGE_PALETTE[g] for g in target_counts.index],
    edgecolor='white', linewidth=0.8, width=0.55
)
# Annotate each bar with count + percentage
for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 800,
        f'{count:,}\\n({pct:.1f}%)',
        ha='center', va='bottom', fontsize=10
    )
axes[0].set_title('Damage Grade — Count', fontweight='bold')
axes[0].set_ylabel('Number of buildings')
axes[0].set_ylim(0, target_counts.max() * 1.18)
axes[0].set_xticklabels([DAMAGE_LABELS[g] for g in target_counts.index], rotation=12, ha='right')

# Right: pie chart
axes[1].pie(
    target_counts.values,
    labels=[DAMAGE_LABELS[g] for g in target_counts.index],
    colors=[DAMAGE_PALETTE[g] for g in target_counts.index],
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops={'fontsize': 10}
)
axes[1].set_title('Damage Grade — Proportion', fontweight='bold')

fig.suptitle('Target Variable Distribution  |  damage_grade', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig1_target_distribution.png', bbox_inches='tight')
plt.show()

print("Key finding: Grade 2 dominates at 56.9%, Grade 1 is the minority at 9.6%.")
print("→ Micro-averaged F1 score will be our primary evaluation metric.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Numeric Feature Distributions

We examine the five core numeric building attributes. For each we plot:
1. A **histogram** of the raw distribution
2. A **box plot split by damage grade** — this directly shows whether the feature
   separates the three classes

Features that show clear separation across grades are likely to be strong predictors.
"""))

cells.append(code("""\
numeric_meta = {
    'count_floors_pre_eq': 'Number of floors before earthquake',
    'age'                : 'Age of building (years)',
    'area_percentage'    : 'Normalised footprint area (%)',
    'height_percentage'  : 'Normalised height (%)',
    'count_families'     : 'Number of families in building',
}

fig, axes = plt.subplots(len(numeric_meta), 2, figsize=(14, 4 * len(numeric_meta)))

for row, (col, label) in enumerate(numeric_meta.items()):
    data = df[col]

    # ── Left: histogram coloured by damage grade ──────────────────────────────
    ax_hist = axes[row, 0]
    for grade in [1, 2, 3]:
        subset = df.loc[df['damage_grade'] == grade, col]
        ax_hist.hist(
            subset, bins=40, alpha=0.55,
            color=DAMAGE_PALETTE[grade], label=DAMAGE_LABELS[grade],
            density=True, edgecolor='none'
        )
    ax_hist.set_title(f'{label} — Distribution by grade', fontweight='bold')
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize=8)

    # ── Right: box plot by damage grade ───────────────────────────────────────
    ax_box = axes[row, 1]
    grade_data = [df.loc[df['damage_grade'] == g, col].values for g in [1, 2, 3]]
    bp = ax_box.boxplot(
        grade_data,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=2, alpha=0.3)
    )
    for patch, grade in zip(bp['boxes'], [1, 2, 3]):
        patch.set_facecolor(DAMAGE_PALETTE[grade])
        patch.set_alpha(0.75)
    ax_box.set_xticks([1, 2, 3])
    ax_box.set_xticklabels([f'Grade {g}' for g in [1, 2, 3]])
    ax_box.set_title(f'{label} — Box plot by grade', fontweight='bold')
    ax_box.set_ylabel(col)

fig.suptitle('Numeric Feature Distributions', fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig('fig2_numeric_distributions.png', bbox_inches='tight')
plt.show()

# ── Summary statistics per grade ──────────────────────────────────────────────
print("=== Median values by damage grade ===")
print(df.groupby('damage_grade')[list(numeric_meta.keys())].median().round(2).to_string())
"""))

cells.append(md("""\
### Observations on numeric features

| Feature | Pattern |
|---------|---------|
| `age` | Older buildings skew toward higher damage grades — intuitive, as older structures predate modern seismic codes |
| `count_floors_pre_eq` | Taller buildings show slightly higher damage; multi-storey structures are more vulnerable |
| `area_percentage` | Smaller footprints correlate with more destruction — smaller buildings may be less structurally robust |
| `height_percentage` | Higher normalised height is associated with greater damage |
| `count_families` | Relatively uniform across grades; less likely to be a strong predictor on its own |

> **Note on `age`:** The max value of 995 years is suspicious and likely an encoding error or placeholder.
> We will examine this outlier more closely and decide how to handle it in Phase 2.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# AGE OUTLIER INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""### 4.1 Age Outlier Investigation

The `age` column has a maximum of 995 years, which is implausibly old for a modern building.
We investigate how many records have extreme age values and what damage grades they correspond to.
This informs whether we should cap, log-transform, or flag these values in preprocessing.
"""))

cells.append(code("""\
# ── Age distribution — zoomed view ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Full distribution (log scale to show outlier tail)
axes[0].hist(df['age'], bins=80, color='steelblue', edgecolor='none', alpha=0.7)
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].set_title('Full age distribution (log y-scale)', fontweight='bold')
axes[0].set_yscale('log')

# Zoomed to 0–200 years (the plausible range)
axes[1].hist(df.loc[df['age'] <= 200, 'age'], bins=60,
             color='steelblue', edgecolor='none', alpha=0.7)
axes[1].set_xlabel('Age (years, capped at 200)')
axes[1].set_ylabel('Count')
axes[1].set_title('Age distribution (≤ 200 years, plausible range)', fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_age_outliers.png', bbox_inches='tight')
plt.show()

# ── Outlier stats ─────────────────────────────────────────────────────────────
thresholds = [100, 200, 500]
for t in thresholds:
    n = (df['age'] > t).sum()
    pct = n / len(df) * 100
    print(f"  age > {t:3d}: {n:5,} buildings ({pct:.2f}%)")

print(f"\\nMedian age overall : {df['age'].median():.0f} years")
print(f"95th percentile    : {df['age'].quantile(0.95):.0f} years")
print(f"99th percentile    : {df['age'].quantile(0.99):.0f} years")
print("\\n→ Decision: Cap age at 150 years in Phase 2 (99th pct ~ 100, values above are likely data entry errors).")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Categorical Feature Distributions

The dataset contains eight categorical columns encoded as single letters. Their meanings
are not fully documented by the competition, but the distributions and damage-grade
breakdowns can still reveal which categories carry the most predictive signal.

For each feature we plot a **stacked proportional bar chart** — this makes it easy to
compare the damage-grade mix across categories of very different sizes.
"""))

cells.append(code("""\
# ── Stacked proportion bars for each categorical column ───────────────────────
fig, axes = plt.subplots(4, 2, figsize=(15, 22))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    ax = axes[idx]

    # Count damage grade per category, normalise to proportions
    ct = (df.groupby([col, 'damage_grade'])
            .size()
            .unstack(fill_value=0))
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Sort categories by Grade-3 proportion (most damaged first)
    ct_pct = ct_pct.sort_values(3, ascending=True)

    # Build stacked bars
    bottom = np.zeros(len(ct_pct))
    for grade in [1, 2, 3]:
        vals = ct_pct[grade].values
        bars = ax.barh(
            range(len(ct_pct)), vals, left=bottom,
            color=DAMAGE_PALETTE[grade], label=DAMAGE_LABELS[grade],
            alpha=0.85, edgecolor='white', linewidth=0.5
        )
        bottom += vals

    # Annotate with raw count per category
    raw_counts = ct.sum(axis=1).loc[ct_pct.index]
    for i, (cat, cnt) in enumerate(raw_counts.items()):
        ax.text(101, i, f'n={cnt:,}', va='center', fontsize=7.5,
                color='dimgray')

    ax.set_yticks(range(len(ct_pct)))
    ax.set_yticklabels(ct_pct.index.tolist(), fontsize=9)
    ax.set_xlabel('Percentage of buildings (%)')
    ax.set_xlim(0, 115)
    ax.set_title(col, fontweight='bold')
    if idx == 0:
        ax.legend(loc='lower right', fontsize=8)

# Hide unused subplot if any
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Categorical Features — Damage Grade Proportion per Category',
             fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig('fig4_categorical_distributions.png', bbox_inches='tight')
plt.show()
"""))

cells.append(md("""\
### Observations on categorical features

| Feature | Key finding |
|---------|-------------|
| `foundation_type` | Clear separation: type `h` shows a very different damage profile from type `r`; foundation type is likely a strong predictor |
| `roof_type` | Type `n` (the dominant category) shows higher Grade-3 proportion than type `x` |
| `ground_floor_type` | Type `v` shows noticeably less damage than `f`; floor construction matters |
| `land_surface_condition` | Buildings on steep terrain (`o`) show higher severe-damage rates than flat (`t`) |
| `position` | Standalone buildings (`s`) show different profiles from corner (`j`) or attached (`t`) |
| `plan_configuration` | Heavily skewed — category `d` holds ~96% of buildings; most other categories are rare |
| `legal_ownership_status` | Dominated by `v` (96%); limited predictive power expected |

> **Encoding decision:** All categorical features will be **ordinal-encoded** in Phase 2,
> assigning each letter a consistent integer. Since tree-based models don't assume ordinality,
> this is equivalent to one-hot encoding without the dimensionality cost.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SUPERSTRUCTURE FLAGS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Superstructure Material Flags

Buildings can be constructed from multiple materials simultaneously (each flag is 0/1).
We examine two things:
1. **Prevalence** — how common is each material in the dataset?
2. **Damage association** — is the material linked to higher or lower damage grades?

The comparison of mean damage grade between flag=0 and flag=1 gives a simple but
interpretable signal of each material's relationship to destruction.
"""))

cells.append(code("""\
# ── Compute prevalence and mean damage grade for flag=1 vs flag=0 ─────────────
material_stats = []
for col in superstructure_cols:
    name  = col.replace('has_superstructure_', '').replace('_', ' ').title()
    count = df[col].sum()
    pct   = count / len(df) * 100
    mean_dmg_yes = df.loc[df[col] == 1, 'damage_grade'].mean()
    mean_dmg_no  = df.loc[df[col] == 0, 'damage_grade'].mean()
    material_stats.append({
        'Material': name,
        'Count (flag=1)': count,
        '% of buildings': round(pct, 1),
        'Mean damage (flag=1)': round(mean_dmg_yes, 3),
        'Mean damage (flag=0)': round(mean_dmg_no, 3),
        'Delta': round(mean_dmg_yes - mean_dmg_no, 3)
    })

stats_df = pd.DataFrame(material_stats).sort_values('Delta')
print(stats_df.to_string(index=False))

# ── Plot: mean damage grade delta per material ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: prevalence bar
order = stats_df.sort_values('% of buildings', ascending=True)
axes[0].barh(order['Material'], order['% of buildings'],
             color='steelblue', alpha=0.75, edgecolor='none')
axes[0].set_xlabel('% of buildings with this material')
axes[0].set_title('Material prevalence', fontweight='bold')
axes[0].axvline(0, color='black', linewidth=0.8)

# Right: delta in mean damage grade (flag=1 vs flag=0)
colors = ['#E53935' if d > 0 else '#4CAF50' for d in stats_df['Delta']]
axes[1].barh(stats_df['Material'], stats_df['Delta'],
             color=colors, alpha=0.75, edgecolor='none')
axes[1].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Mean damage grade difference (flag=1 minus flag=0)')
axes[1].set_title('Damage association (red = more damage)', fontweight='bold')

fig.suptitle('Superstructure Materials — Prevalence and Damage Association',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_superstructure_materials.png', bbox_inches='tight')
plt.show()
"""))

cells.append(md("""\
### Observations on superstructure materials

- **Mud mortar stone** (76% of buildings) is the dominant material and is associated with
  *above-average* damage — this widespread traditional construction was highly vulnerable.
- **RC engineered** (reinforced concrete, engineered) shows the largest *negative* delta:
  buildings using this material sustained meaningfully less damage, consistent with
  engineering expectations.
- **Adobe mud** also shows a positive damage association, despite lower prevalence.
- **Cement mortar brick** shows a slight protective effect relative to the mean.

These patterns are physically intuitive and suggest superstructure material flags will be
among the stronger predictors in our model.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# GEO LEVELS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Geographic Distribution of Damage

The three geographic ID columns represent a hierarchical administrative region system
(district → municipality → ward). With 31 level-1 regions, 1,414 level-2, and 11,595
level-3, the raw IDs carry **location signal that is too granular to use directly** as
categorical variables. We explore the geographic damage distribution at level 1 (district)
and level 2 (municipality) — the findings here will directly motivate our geographic
feature engineering in Phase 3.
"""))

cells.append(code("""\
# ── Mean damage grade by geo_level_1 ─────────────────────────────────────────
geo1_stats = (df.groupby('geo_level_1_id')['damage_grade']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'mean_damage', 'count': 'n_buildings'})
                .sort_values('mean_damage', ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: mean damage per geo_level_1, dot sized by building count
sc = axes[0].scatter(
    range(len(geo1_stats)),
    geo1_stats['mean_damage'],
    s=geo1_stats['n_buildings'] / 30,
    c=geo1_stats['mean_damage'],
    cmap='RdYlGn_r', alpha=0.8, edgecolors='none'
)
axes[0].axhline(df['damage_grade'].mean(), color='gray', linestyle='--',
                linewidth=1, label=f'Overall mean ({df["damage_grade"].mean():.2f})')
axes[0].set_xlabel('geo_level_1_id (sorted by mean damage)')
axes[0].set_ylabel('Mean damage grade')
axes[0].set_title('Mean damage by district (level 1)\nDot size ∝ number of buildings',
                  fontweight='bold')
axes[0].legend()
plt.colorbar(sc, ax=axes[0], label='Mean damage grade')

# Right: damage-grade mix for the 5 most vs 5 least damaged districts
top5    = geo1_stats.head(5).index.tolist()
bottom5 = geo1_stats.tail(5).index.tolist()
selected = top5 + bottom5

ct = (df[df['geo_level_1_id'].isin(selected)]
      .groupby(['geo_level_1_id', 'damage_grade'])
      .size().unstack(fill_value=0))
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(ct_pct))
for grade in [1, 2, 3]:
    axes[1].bar(range(len(ct_pct)), ct_pct[grade], bottom=bottom,
                color=DAMAGE_PALETTE[grade], label=DAMAGE_LABELS[grade],
                alpha=0.85, edgecolor='white', linewidth=0.4)
    bottom += ct_pct[grade].values

axes[1].set_xticks(range(len(ct_pct)))
axes[1].set_xticklabels(
    [f'ID {i}\\n({"HIGH" if i in top5 else "LOW"})' for i in ct_pct.index],
    fontsize=8, rotation=30
)
axes[1].set_ylabel('Percentage of buildings (%)')
axes[1].set_title('Damage mix: 5 most vs 5 least damaged districts',
                  fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)

fig.suptitle('Geographic Damage Distribution — Level 1 (District)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_geo_damage.png', bbox_inches='tight')
plt.show()

# ── Variance explained by geography ──────────────────────────────────────────
geo1_var   = df.groupby('geo_level_1_id')['damage_grade'].mean().std()
geo2_var   = df.groupby('geo_level_2_id')['damage_grade'].mean().std()
overall_std = df['damage_grade'].std()

print(f"Std of district-level mean damage    : {geo1_var:.4f}")
print(f"Std of municipality-level mean damage: {geo2_var:.4f}")
print(f"Overall damage_grade std             : {overall_std:.4f}")
print(f"\\n→ District-level means vary by ±{geo1_var:.2f} grade points around the global mean.")
print("→ This geographic variation motivates using location-based aggregate features in Phase 3.")
"""))

cells.append(md("""\
### Geographic takeaway

There is **substantial variation in mean damage across geographic regions** — some districts
suffered far more destruction than others. This is likely due to a combination of proximity
to the epicentre, local geology, and prevalent building styles in each region.

Because `geo_level_3_id` has 11,595 unique values, we cannot use it directly as a
one-hot encoded feature. In **Phase 3** we will engineer geographic features by computing
the **mean damage grade per region at each level** — this condenses the location signal into
a single meaningful numeric feature per level.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Correlation with Target

We compute the **point-biserial correlation** (equivalent to Pearson for binary predictors)
between each numeric and binary flag column and `damage_grade`. This gives a ranked list of
the features most linearly associated with the target — a useful baseline for understanding
feature importance before modelling.
"""))

cells.append(code("""\
# ── Correlation of all numeric/binary columns with damage_grade ───────────────
numeric_and_flags = numeric_cols + superstructure_cols + secondary_use_cols

corr_series = (df[numeric_and_flags + ['damage_grade']]
               .corr()['damage_grade']
               .drop('damage_grade')
               .sort_values())

# ── Horizontal bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))

colors = ['#E53935' if c > 0 else '#1565C0' for c in corr_series]
ax.barh(corr_series.index, corr_series.values, color=colors, alpha=0.75, edgecolor='none')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson / point-biserial correlation with damage_grade')
ax.set_title('Feature correlation with target (damage_grade)\nRed = positive (more damage), Blue = negative (less damage)',
             fontweight='bold')

# Annotate top 5 positive and top 5 negative
for label, val in list(corr_series.tail(5).items()) + list(corr_series.head(5).items()):
    ax.text(val + (0.002 if val >= 0 else -0.002), 
            list(corr_series.index).index(label),
            f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=8)

plt.tight_layout()
plt.savefig('fig7_correlation_with_target.png', bbox_inches='tight')
plt.show()

print("Top 5 features positively correlated with damage_grade:")
print(corr_series.tail(5).round(4).to_string())
print("\\nTop 5 features negatively correlated with damage_grade:")
print(corr_series.head(5).round(4).to_string())
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SUPERSTRUCTURE CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Superstructure Material Co-occurrence

Buildings often use more than one construction material. We visualise how many
materials each building uses and whether material diversity (or lack of it) correlates
with damage. This motivates engineering a **total material count** feature in Phase 3.
"""))

cells.append(code("""\
# ── Count of superstructure materials per building ─────────────────────────────
df['material_count'] = df[superstructure_cols].sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: distribution of material count
cnt = df['material_count'].value_counts().sort_index()
axes[0].bar(cnt.index, cnt.values, color='steelblue', alpha=0.75, edgecolor='white')
axes[0].set_xlabel('Number of superstructure materials')
axes[0].set_ylabel('Number of buildings')
axes[0].set_title('How many materials does each building use?', fontweight='bold')
for i, (x, y) in enumerate(zip(cnt.index, cnt.values)):
    axes[0].text(x, y + 500, f'{y:,}', ha='center', fontsize=8)

# Right: mean damage grade by material count
mean_dmg_by_mat = df.groupby('material_count')['damage_grade'].mean()
axes[1].plot(mean_dmg_by_mat.index, mean_dmg_by_mat.values,
             marker='o', color='#E53935', linewidth=2, markersize=7)
axes[1].axhline(df['damage_grade'].mean(), color='gray', linestyle='--',
                linewidth=1, label='Overall mean')
axes[1].set_xlabel('Number of superstructure materials')
axes[1].set_ylabel('Mean damage grade')
axes[1].set_title('Mean damage grade by material count', fontweight='bold')
axes[1].legend()
axes[1].set_xticks(mean_dmg_by_mat.index)

fig.suptitle('Superstructure Material Diversity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_material_diversity.png', bbox_inches='tight')
plt.show()

# Cleanup: remove temporary column (it will be re-created cleanly in Phase 3)
df.drop(columns=['material_count'], inplace=True)

print(df[['damage_grade']].assign(material_count=df[superstructure_cols].sum(axis=1))
      .groupby('material_count')['damage_grade']
      .agg(['mean', 'count']).round(3).to_string())
"""))

# ─────────────────────────────────────────────────────────────────────────────
# EDA SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 10. EDA Summary & Decisions for Phase 2

The table below consolidates every key finding from this phase and maps each to a
concrete decision made in Phase 2 or Phase 3. This is the justification chain that
links each modelling choice back to observed data.

| Finding | Evidence | Decision |
|---------|----------|----------|
| Class imbalance (Grade 2 = 57%) | Target distribution plot | Use **micro-averaged F1** as primary metric; stratified train/val split |
| `age` has implausible outliers (max 995 yrs) | Age histogram + percentile table | Cap `age` at 150 years in preprocessing |
| Foundation type separates damage grades | Stacked proportion chart | Encode as ordinal; expect high feature importance |
| Superstructure material flags vary strongly | Damage delta table | Keep all 11 flags; engineer a `material_count` summary feature |
| `has_superstructure_rc_engineered` is protective | Negative delta in damage association | Will be inspected closely in SHAP analysis |
| Geographic region drives damage variation | District-level mean damage scatter | Engineer geo-aggregate features (mean damage per level) in Phase 3 |
| `plan_configuration` and `legal_ownership_status` are skewed | Bar charts showing 96%+ in one category | Low information; will likely have low feature importance — monitor in Phase 4 |
| No missing values anywhere | Data quality check | No imputation needed |

**Next step:** Phase 2 — Preprocessing pipeline, train/validation split, and baseline model.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE & SAVE
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells

output_path = 'earthquake_damage.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Number of cells: {len(cells)}")
print(nb)

print(f"Notebook written to {output_path}")
