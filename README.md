# EventSaathi — Event-Driven Congestion Prediction System

> **Flipkart Grid 6.0 | Round 2 | Problem 3**
> A full-stack, ML-powered system to forecast traffic congestion duration, predict road closure probability, and generate intelligent resource deployment plans for planned and unplanned urban events.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Dataset & Initial Data Analysis](#dataset--initial-data-analysis)
4. [Experimental Phase — `fg2_exp3_2.ipynb`](#experimental-phase--fg2_exp3_2ipynb)
5. [Refined Experimentation — `fg2_exp3_2-Copy1.ipynb`](#refined-experimentation--fg2_exp3_2-copy1ipynb)
6. [Final Clean Training — `fg2_clean1.ipynb` (Task A: Duration Regression)](#final-clean-training--fg2_clean1ipynb-task-a-duration-regression)
7. [Final Clean Training — `fg2_clean2.ipynb` (Task B: Road Closure Classification)](#final-clean-training--fg2_clean2ipynb-task-b-road-closure-classification)
8. [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
9. [Model Architecture & Selection](#model-architecture--selection)
10. [Backend: FastAPI Service](#backend-fastapi-service)
11. [Frontend: React Dashboard](#frontend-react-dashboard)
13. [How to Run](#how-to-run)
14. [Running the Notebooks](#running-the-notebooks)
15. [API Reference](#api-reference)

---

## Problem Statement

Urban traffic systems in cities like Bengaluru face a recurring crisis: **events — both planned and unplanned — create localized congestion breakdowns that the system is unprepared for.** Political rallies, religious processions, tree falls, vehicle breakdowns, waterlogging, and construction activities can paralyze entire zones for hours.

The core operational gaps identified:

- **Event impact is not quantified in advance.** Traffic managers have no systematic estimate of how long a vehicle breakdown at a Central Zone junction during morning peak will last.
- **Resource deployment is experience-driven.** Decisions about how many police officers to deploy, how many barricades to set up, and whether to activate diversions are made on intuition.
- **There is no post-event learning system.** Historical data from thousands of incidents lies dormant, never used to improve future responses.

**The challenge:** *How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?*

---

## Solution Overview

EventSaathi is a three-layer solution:

**Layer 1 — ML Prediction Engine (Two Models)**
- **Task A (Duration Regression):** An XGBoost model trained on log-transformed incident duration, predicting how long an event will last in minutes. Uses 42 engineered features including temporal cyclical encodings, geographic bins, target-encoded location features, and interaction terms.
- **Task B (Road Closure Classification):** A separate XGBoost classifier predicting the probability that an incident will require a road closure. Trained with leakage-free features (closure-related flags excluded) and class imbalance correction via `scale_pos_weight`.

**Layer 2 — FastAPI Backend**
A Python REST API that loads the trained `.pkl` model artifacts, reconstructs the feature row from user input at inference time (replicating the exact same encoding pipeline used during training), and returns a structured JSON response including duration, severity, road closure probability, resource recommendations, similar historical events, and feature importances.

**Layer 3 — React Frontend Dashboard**
A GitHub-dark-themed single-page application with a two-panel layout: an input form on the left and a four-tab results dashboard (Prediction, Resources, Explainability, Similar Events) on the right.

---

## Dataset & Initial Data Analysis

The dataset is a real Bengaluru traffic incident log exported from the city's operational system. It contains **8,173 rows** and roughly 45 columns, with massive missingness across most columns.

### Missing Data Summary (Selected Critical Columns)

| Column | Missing Values | Percentage |
|---|---|---|
| `comment`, `map_file`, `meta_data` | 8,173 | 100.0% |
| `direction` | 8,130 | 99.5% |
| `resolved_at_*` (address/lat/lon/id/datetime) | 8,099 | 99.1% |
| `assigned_to_police_id` | 8,045 | 98.4% |
| `age_of_truck`, `reason_breakdown`, `cargo_material` | 7,897 | 96.6% |
| `end_datetime` | 7,683 | 94.0% |
| `end_address` | 7,486 | 91.6% |
| `junction` | 5,663 | 69.3% |
| `closed_datetime` | 5,032 | 61.6% |
| `zone`, `gba_identifier` | 4,729 | 57.9% |
| `veh_no`, `veh_type` | ~3,286 | 40.2% |
| `endlatitude`, `endlongitude` | 169 | 2.1% |
| `corridor` | 20 | 0.2% |
| `event_cause`, `latitude`, `longitude`, `event_type`, `status`, `start_datetime`, `requires_road_closure` | 0 | 0.0% |

### Key Observations from the Data

**The target variable had to be engineered.** There was no direct "duration" column. Instead, the incident end time had to be constructed by taking `closed_datetime`, falling back to `resolved_datetime` when `closed_datetime` was missing (61.6% of rows). The duration was then computed as `(end_ts - start_datetime).dt.total_seconds() / 60`.

**The `junction` column was 69.3% missing.** This is a highly predictive location feature, but it cannot be used directly. The solution was to fill `Unknown` and target-encode it against duration — this way even `Unknown` gets a meaningful mean duration value rather than being dropped.

**`zone` was 57.9% missing**, creating the same problem at a geographic level. Same strategy: fill `Unknown`, then target-encode.

**`veh_type` was 40.2% missing.** Vehicle type matters (a heavy truck breakdown lasts longer than a private car breakdown), but over 3,000 rows had no vehicle type recorded. Fill with `Unknown` and label-encode.

**Columns like `comment`, `map_file`, `direction`, `resolved_at_*` were essentially empty** (>99% missing) and had to be dropped entirely from consideration.

**The core usable features:** `event_cause`, `event_type`, `latitude`, `longitude`, `start_datetime`, `requires_road_closure`, `priority`, `zone` (after fill), `junction` (after fill), `corridor`, `veh_type` (after fill), `endlatitude`/`endlongitude` (as a binary flag for span).

### Row Filtering

After computing duration, four filters were applied to produce a clean working set:
1. `duration_mins` must not be null (need both start and an end timestamp)
2. `duration_mins > 0` (eliminate data entry errors with zero or negative durations)
3. `duration_mins < 1440` (eliminate events logged as lasting more than 24 hours — likely unclosed tickets rather than real events)
4. `priority` must not be null (required feature)
5. `address` must not be null (eliminates records with fundamentally broken location data)

This filtering reduced the 8,173 rows to a clean working dataset of approximately 2,000–3,000 usable rows, which is realistic for an operational city incident system spanning a few months of data.

### Severity Label Construction

Severity was derived from duration using domain-informed bins:

| Duration Range | Severity Label | Encoded Value |
|---|---|---|
| 0 – 30 mins | Low | 0 |
| 30 – 90 mins | Medium | 1 |
| 90 – 240 mins | High | 2 |
| 240 – 1440 mins | Critical | 3 |

These thresholds were chosen to reflect real operational significance: a 20-minute breakdown barely disrupts flow, a 2-hour accident needs significant resource deployment, and a 4+ hour event (procession, construction) warrants full diversion activation.

---

## Experimental Phase — `fg2_exp3_2.ipynb`

This notebook is the **first serious attempt** at modeling after initial data exploration. It established the baseline pipeline and ran a broad model comparison across three tasks simultaneously.

### Pipeline Setup

```
Raw CSV → datetime parsing → duration computation → severity labeling
       → row filtering → categorical fill → temporal feature extraction
       → geo binning → label encoding → target encoding → train/test split
```

### Feature Set (Version 1)

The initial feature set was a flat 24-column vector:

```python
FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak', 'is_night',
    'latitude', 'longitude', 'lat_bin', 'lon_bin', 'has_end_location',
    'event_type_enc', 'event_cause_enc',
    'veh_type_enc', 'status_enc',
    'zone_tenc', 'zone_known',
    'junction_tenc', 'junction_known',
    'corridor_tenc', 'corridor_known',
    'gba_identifier_tenc', 'gba_identifier_known',
    'priority_enc',
]
```

Note that `status_enc` (the current incident status field) was included here — this was later identified as a potential leakage source and dropped in the final clean version. Similarly, `gba_identifier` (a geographic block identifier) was target-encoded, but it was later consolidated because it was highly correlated with `zone` and added complexity without clear benefit.

### Task A: Duration Regression

Four models were compared head-to-head on a single 80/20 train-test split:

**Ridge Regression (Baseline)**
A linear model with L2 regularization, run through a `StandardScaler → Ridge` pipeline. Since duration is a mix of linear and non-linear signals (a `junction_tenc` value of 120 doesn't mean twice the duration of 60 in any simple sense), Ridge was expected to underperform but served as a sanity check.

**Random Forest Regressor**
`RandomForestRegressor(n_estimators=200, max_depth=10)`. Tree-based model that handles non-linear relationships and missing values naturally. The `n_jobs=-1` flag was used throughout for parallel training.

**XGBoost Regressor**
```python
xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=30, eval_metric='mae'
)
```
Gradient boosting with early stopping on the validation set to prevent overfitting. Subsampling of both rows (0.8) and features (0.8) per tree adds implicit regularization.

**LightGBM Regressor**
```python
lgb.LGBMRegressor(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)
```
Histogram-based gradient boosting — faster than XGBoost on this dataset size, slightly different inductive bias.

Target for Task A was raw `duration_mins` (not log-transformed at this stage — that came in the clean version).

### Task B: Road Closure Binary Classification

The key insight here was **leakage prevention**. The `requires_road_closure` column (the target) was correlated with `has_end_location` — incidents that span two geographic points almost always require a road closure because the blocked segment is explicitly delineated. Using `has_end_location` as a feature when predicting `requires_road_closure` would inflate metrics through data leakage.

```python
FEATURE_COLS_B = [
    c for c in FEATURE_COLS
    if c not in ('requires_road_closure', 'has_end_location')
]
```

Three classifiers were tested: Logistic Regression (baseline), Random Forest, and XGBoost. XGBoost was configured with `scale_pos_weight = neg/pos` to handle the class imbalance (road closures are the minority class), and used `eval_metric='aucpr'` — area under the precision-recall curve — which is more informative than AUC-ROC for imbalanced classification.

### Task C: Priority Multi-class Classification

A separate multi-class task was tested to see if the input features could predict incident `priority` (low/medium/high/critical). The `priority_enc` feature was carefully excluded from `FEATURE_COLS_C` to avoid leakage when predicting the priority label.

This task was ultimately **dropped from the final production system**. The reason: priority in the dataset was assigned by operators at intake, not derived from objective event characteristics. The model had reasonable F1 but the task was not operationally useful — you already know the priority when you're filling in the form.

### Hyperparameter Tuning (Experimental Notebook)

After the baseline comparisons, the experimental notebook ran an extensive tuning pass:

**O1: RandomizedSearchCV (80 iterations)**
A broad random sweep over XGBoost hyperparameters: `n_estimators` (200–1000), `max_depth` (3–10), `learning_rate` (0.01–0.30), `subsample` (0.5–1.0), `colsample_bytree` (0.5–1.0), `min_child_weight` (1–10), `gamma` (0–0.5), `reg_alpha` (0–1.0), `reg_lambda` (0.5–2.0). Scoring was `neg_mean_absolute_error` over 5-fold CV.

**O2: Optuna Bayesian Optimization (100 trials)**
Replaced the random search with Optuna's TPE (Tree-structured Parzen Estimator) sampler — a smarter search that learns from previous trials to focus on promising hyperparameter regions. Also added a `MedianPruner` to kill unpromising trials early. The Optuna search ran 100 trials and converged on a much better configuration than the random search in the same compute budget.

**O3: BaggingRegressor over XGBoost**
Wrapped the best Optuna-tuned XGBoost in a `BaggingRegressor(n_estimators=10, max_samples=0.8, max_features=0.9)` to test whether ensemble variance reduction on top of boosting helped. Result: marginal improvement in OOF MAE, but significant increase in inference time.

**O4: Boosting Variants**
Compared tuned XGBoost, tuned LightGBM (separate 60-trial Optuna run for LGB), and sklearn's `GradientBoostingRegressor` side by side with full CV metrics (MAE, RMSE, R²).

**O5: Blending and Stacking**
Generated out-of-fold predictions from XGBoost, LightGBM, and Random Forest, then:
- Optimal weighted blend: Grid search over (w_xgb, w_lgb, w_rf) with step 0.1, minimizing OOF MAE.
- StackingRegressor with Ridge meta-learner
- StackingRegressor with Lasso meta-learner

The stacking improved MAE marginally over the best single model but introduced significant deployment complexity (three models needed at inference instead of one).

**Decision from experimentation:** XGBoost with Optuna-tuned hyperparameters, trained on a richer feature set, was the practical winner. The stacking improvement was not worth the inference complexity for an operational API.

---

## Refined Experimentation — `fg2_exp3_2-Copy1.ipynb`

This notebook is a **cleaned-up rerun of the experimental notebook** with better inline documentation of each step. The code structure and pipeline were the same, but this version added:

- Explicit comments on *why* each feature was included (e.g., explaining that `_known` flags capture whether a location field was originally missing vs. filled)
- Better-organized feature column separation (FEATURE_COLS_A, B, C defined more cleanly)
- The same model comparisons run again to confirm reproducibility of results

The key refinements documented here that fed into the clean version:

1. `status_enc` was noted as a possible leakage risk — incident status changes as the event progresses, so using it at prediction time (when the event has just started) would require it to always be "open," making it uninformative. Drop it.
2. `gba_identifier` was noted as a near-duplicate of `zone` with 57.9% missing — the additional target encoding added noise, not signal. Drop it.
3. The target variable `duration_mins` exhibited right-skewed distribution with heavy outliers. Direct regression on raw minutes was penalizing short-duration predictions. A log transformation was recommended for the clean version.
4. Interaction features were conceptually tested: a `procession` at peak hour is qualitatively different from a `procession` at midnight. The model should be able to learn this, but a direct product feature (`cause_enc × is_peak`) makes this interaction explicit and easier for gradient boosting to capture with fewer splits.

---

## Final Clean Training — `fg2_clean1.ipynb` (Task A: Duration Regression)

This notebook implements the **production-ready duration regression model** from scratch, incorporating all lessons from the experimental phase.

### Step 1: Data Preprocessing

```python
for col in ['closed_datetime', 'start_datetime', 'resolved_datetime', 'created_date']:
    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

df['end_ts']        = df['closed_datetime'].fillna(df['resolved_datetime'])
df['duration_mins'] = (df['end_ts'] - df['start_datetime']).dt.total_seconds() / 60
```

**Why UTC-aware parsing:** The raw timestamps in the dataset had timezone information that needed to be handled consistently. `errors='coerce'` converts any unparseable string to NaT rather than crashing.

**Why fallback to `resolved_datetime`:** `closed_datetime` was 61.6% missing, but `resolved_datetime` was only 99.1% missing — confusingly, many rows had neither. The fallback chain ensures maximum row retention: a row with `resolved_datetime` but no `closed_datetime` still gets a valid duration.

Row filters were applied identically to the experimental phase, producing the same clean working set.

### Step 2: Expanded Temporal Features

The clean version significantly expanded temporal engineering beyond the 3-feature set (hour, day_of_week, month) used in experiments:

**Granular time-of-day flags:**
```python
df_clean['is_peak_am'] = df_clean['hour'].isin([7, 8, 9]).astype(int)
df_clean['is_peak_pm'] = df_clean['hour'].isin([17, 18, 19]).astype(int)
df_clean['is_night']   = df_clean['hour'].isin([22, 23, 0, 1, 2]).astype(int)
df_clean['is_monday']  = (df_clean['day_of_week'] == 0).astype(int)
df_clean['is_friday']  = (df_clean['day_of_week'] == 4).astype(int)
```

Why Monday and Friday specifically? Monday incidents are often more severe because the city returns to full activity after the weekend — accumulated road damage, construction resumption, higher vehicle counts. Friday evenings are the worst PM peak of the week due to combined commuter and leisure traffic.

**Calendar granularity:**
```python
df_clean['week_of_year'] = df_clean['start_datetime'].dt.isocalendar().week.astype(int)
df_clean['quarter']      = df_clean['start_datetime'].dt.quarter
```

Week-of-year captures seasonal patterns (festival seasons, monsoon months) that month alone misses. Quarter captures broader seasonality.

**Cyclical encoding of time features:**
```python
df_clean['hour_sin']  = np.sin(2 * np.pi * df_clean['hour'] / 24)
df_clean['hour_cos']  = np.cos(2 * np.pi * df_clean['hour'] / 24)
df_clean['dow_sin']   = np.sin(2 * np.pi * df_clean['day_of_week'] / 7)
df_clean['dow_cos']   = np.cos(2 * np.pi * df_clean['day_of_week'] / 7)
df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
```

This is the most important temporal improvement. Tree-based models see hour=23 and hour=0 as very different numbers (23 and 0), but they represent adjacent times. The cyclical (sin/cos) encoding captures the circular nature of time: hour 23 and hour 0 are close in sine-cosine space even though their integer values are far apart. The same logic applies to days of week and months of year.

### Step 3: Geographic Features

```python
df_clean['lat_bin'] = pd.cut(df_clean['latitude'],  bins=8, labels=False)
df_clean['lon_bin'] = pd.cut(df_clean['longitude'], bins=8, labels=False)
df_clean['has_end_location'] = (df_clean['endlatitude'].fillna(0) != 0).astype(int)
df_clean['dist_from_center'] = np.sqrt(
    (df_clean['latitude'] - 12.9716)**2 + (df_clean['longitude'] - 77.5946)**2)
```

**Binning latitude/longitude:** Raw coordinates are too specific for a model to learn from (it would see a specific junction but never generalize). Dividing Bengaluru into an 8×8 grid of geographic cells allows the model to learn zone-level patterns from coordinate data alone.

**Distance from city center (12.9716°N, 77.5946°E — Bengaluru centroid):** Central zones typically have more complex traffic networks, more alternate routes, and faster clearance times. Peripheral areas may have slower emergency response and simpler networks. This single scalar captures this gradient without needing to hardcode zone logic.

**`has_end_location`:** When an incident spans two GPS points (a road-blocking procession, a multi-vehicle accident across an intersection), the incident almost always lasts longer because more area is affected. This binary flag is a strong predictor of duration and road closure.

### Step 4: Categorical Encoding

**Label Encoding (for low-cardinality categoricals):**
```python
LE_COLS = ['event_type', 'event_cause', 'veh_type']
```
These three columns had 2, ~12, and ~11 unique values respectively. Label encoding assigns a consistent integer to each category. The encoders were saved (`le_encoders` dict) so inference can apply the exact same mapping.

`status_enc` was **deliberately excluded** from the clean version. During inference, `status` would always be "OPEN" (the incident just started), making it an uninformative constant feature. Including it in training would either cause leakage (if status changes during the event were recorded) or add noise.

**Priority Ordinal Encoding:**
```python
priority_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
df_clean['priority_enc'] = df_clean['priority'].str.lower().map(priority_map).fillna(1)
```

Priority is ordinal — `critical > high > medium > low` in a meaningful mathematical sense. Using integers 0-3 (not arbitrary label encoding integers) preserves this ordering for the gradient boosting splits.

### Step 5: Target Encoding for High-Cardinality Columns

```python
TENC_COLS = ['zone', 'junction', 'corridor', 'event_cause']
global_tenc = {}
for col in TENC_COLS:
    mean_map  = df_clean.groupby(col)['duration_mins'].mean()
    count_map = df_clean[col].value_counts()
    global_tenc[col] = mean_map.to_dict()
    df_clean[f'{col}_tenc'] = df_clean[col].map(mean_map).fillna(df_clean['duration_mins'].mean())
    df_clean[f'{col}_count'] = df_clean[col].map(count_map).fillna(0)
```

Target encoding replaces a categorical value with the mean duration for incidents of that category. For `junction`, this means each of the 229 unique junctions gets replaced by the average duration of past incidents at that junction — a scalar that carries rich operational meaning.

Why also encode `event_cause` via target encoding when it's already label-encoded? Because label encoding only preserves the category identity (e.g., `procession = 7`), while target encoding preserves its historical impact on duration (e.g., `procession → 183 minutes on average`). Having both encodings lets the model use both types of signal.

The `_count` companion feature captures how many historical incidents fell into each category. Categories with very few examples (e.g., `corridor="Outer Ring Road, MG Road"` with only 5 incidents) should be treated with less confidence than well-populated categories. The count feature lets the model learn this implicitly.

The `global_tenc` dict was saved alongside the model for use at inference time — this ensures that when `main.py` receives a zone name like `"Central Zone 1"`, it looks up the same mean duration value the model was trained with.

### Step 6: Interaction Features

```python
df_clean['cause_x_peak']    = df_clean['event_cause_enc'] * df_clean['is_peak']
df_clean['cause_x_weekend'] = df_clean['event_cause_enc'] * df_clean['is_weekend']
df_clean['cause_x_closure'] = df_clean['event_cause_enc'] * df_clean['requires_road_closure']
df_clean['zone_x_peak']     = df_clean['zone_tenc'] * df_clean['is_peak']
df_clean['closure_x_end']   = df_clean['requires_road_closure'] * df_clean['has_end_location']
```

Tree models can in principle learn interactions by splitting on multiple features across tree depth, but explicit product features help when:
1. The dataset is small (limited data to discover interactions from deep splits alone)
2. The interaction is conceptually important and should be weighted more heavily

**Why `cause_x_peak`:** A vehicle breakdown at 10 AM vs 8 AM is dramatically different — the AM peak at 8 AM has 3× the vehicle density, so the same breakdown blocks far more cars and takes longer to clear because rescue vehicles can't reach the scene quickly.

**Why `zone_x_peak`:** A zone with historically high congestion (high `zone_tenc`) during peak hours is disproportionately worse. The multiplicative interaction captures the compounding effect.

**Why `closure_x_end`:** A road closure combined with a multi-point incident (spanning two locations) is the worst-case operational scenario — it almost always indicates a Critical event.

### Step 7: Log Transformation of Target

```python
df_clean['log_duration'] = np.log1p(df_clean['duration_mins'])
y_log = df_clean['log_duration'].values
```

Duration is right-skewed. Most incidents last 30–90 minutes, but some last 6–8 hours. Without log transformation, the model would be dominated by outlier high-duration events. `log1p(x) = log(1 + x)` is used instead of `log(x)` to handle the edge case of duration = 0 safely (though filtering ensures this doesn't happen, it's good practice).

At prediction time, `np.expm1(pred)` is used to invert the transformation: `expm1(x) = e^x - 1`.

### Step 8: Cross-Validation Strategy

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

The clean version uses **5-fold cross-validation with OOF (out-of-fold) predictions** rather than a single train/test split. This is critical for small datasets: with ~2,000 rows, a single 80/20 split means only ~400 test rows, which gives a noisy MAE estimate. 5-fold CV uses all data for both training and evaluation, producing a much more reliable performance estimate.

OOF predictions are assembled across folds and the MAE/RMSE/R² are computed on the concatenated OOF predictions against all labels — this is equivalent to a leave-one-out estimate and is the most honest metric.

### Step 9: Optuna Hyperparameter Optimization (80 Trials)

```python
def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0),
    }
    maes = []
    for tr_idx, va_idx in kf.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr_idx], y_log[tr_idx],
              eval_set=[(X.iloc[va_idx], y_log[va_idx])], verbose=False)
        pred = np.expm1(m.predict(X.iloc[va_idx]))
        maes.append(mean_absolute_error(y_raw[va_idx], pred))
    return np.mean(maes)
```

The objective function trains on log-duration but evaluates MAE on raw minutes — this is deliberate. We want to minimize error in minutes (what the user sees), but we train on logs to avoid the outlier-dominated gradient problem.

Key tuning decisions:
- `learning_rate` searched on a log scale (0.01–0.3) because small differences matter more at low learning rates
- `reg_alpha` and `reg_lambda` provide L1 and L2 regularization — important for preventing overfitting on a small dataset
- `gamma` is the minimum split loss required to add a new tree node — acts as a structural regularizer

### Step 10: Final Model Training & Saving

```python
final_model = xgb.XGBRegressor(**best_params, verbosity=0, n_jobs=-1, random_state=42)
final_model.fit(X, y_log)  # Train on ALL data with best hyperparams

joblib.dump(final_model,  'models_v3/xgb_clean.pkl')
joblib.dump(FEATURE_COLS, 'models_v3/feature_cols_clean.pkl')
joblib.dump(global_tenc,  'models_v3/global_tenc_clean.pkl')
joblib.dump(le_encoders,  'models_v3/label_encoders_clean.pkl')
joblib.dump(best_params,  'models_v3/best_params_clean.pkl')
```

The final model is retrained on **all available data** (not just the training fold) using the best hyperparameters found by Optuna. This maximizes the amount of data the production model has seen.

Five artifacts are saved:
- `xgb_clean.pkl` — the trained model weights
- `feature_cols_clean.pkl` — the ordered list of feature names (critical for `row_to_array()`)
- `global_tenc_clean.pkl` — the target encoding dictionaries
- `label_encoders_clean.pkl` — the fitted `LabelEncoder` objects
- `best_params_clean.pkl` — the Optuna best parameters (for reference/retraining)

### Sanity Checks

Five test cases were constructed with manually specified feature values to verify the model's predictions make intuitive sense:

| Scenario | Expected | Model Output |
|---|---|---|
| Minor breakdown, 2pm, no closure | Low, ~25-40 mins | Low |
| Procession, 8am peak, road closed | High/Critical, 2-4 hrs | High/Critical |
| Tree fall, 8am peak, road closed | High, 1-3 hrs | High |
| Protest, Friday evening peak | Critical | Critical |
| Water logging, night, no closure | Medium, ~45-70 mins | Medium |

The sanity check was designed with `make_proper_test()` using global feature means as defaults (not arbitrary -999 values), which gives a more realistic baseline.

---

## Final Clean Training — `fg2_clean2.ipynb` (Task B: Road Closure Classification)

This notebook trains the road closure probability model. It **deliberately reuses the encoders** saved by `fg2_clean1.ipynb` to ensure perfect consistency between the two models.

### Why a Separate Model?

Road closure is a binary outcome that needs its own prediction separately from duration. A long incident (240 mins) doesn't always require closure (e.g., slow construction work), and a short incident might (e.g., a procession blocking a main road for 45 minutes). The two signals are correlated but not identical.

### Reusing Task A's Encoders

```python
MODEL_DIR = 'models_v3'
le_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders_clean.pkl'))
global_tenc = joblib.load(os.path.join(MODEL_DIR, 'global_tenc_clean.pkl'))
```

This is crucial for production correctness. If Task B trained its own label encoder, `event_cause='procession'` might get encoded as integer 4 in Task A and integer 7 in Task B. At inference time, `main.py` computes one feature row and uses it for both models — both models must agree on the encoding.

```python
def safe_transform(le, series):
    known = set(le.classes_)
    vals  = series.astype(str).where(series.astype(str).isin(known), le.classes_[0])
    return le.transform(vals)
```

`safe_transform` handles unseen categories gracefully by mapping them to the first known class — matching exactly what `main.py`'s `safe_le()` function does at inference.

### Leakage-Free Feature Set for Task B

The critical insight: `requires_road_closure`, `has_end_location`, `cause_x_closure`, and `closure_x_end` are all directly derived from or highly correlated with the target variable.

```python
LEAKY_FOR_B = {'requires_road_closure', 'has_end_location', 'cause_x_closure', 'closure_x_end'}
FEATURE_COLS_B = [c for c in FEATURE_COLS_A if c not in LEAKY_FOR_B]
```

A correlation check was added as a sanity check:
- `has_end_location` corr with target: very high (incidents spanning two points almost always require closure)
- `cause_x_closure` and `closure_x_end` obviously contain the target variable directly

Including any of these would give unrealistically high AUC on evaluation (model would learn to use the target as a feature), but fail completely in production (you don't know if closure is required *before* you're trying to predict it).

### Class Imbalance Handling

Road closures are the minority class (not every incident requires a closure). XGBoost handles this via:

```python
neg, pos = (y_b == 0).sum(), (y_b == 1).sum()
spw_global = neg / pos
# Used as: scale_pos_weight=spw_global in XGBClassifier
```

`scale_pos_weight = neg/pos` tells XGBoost to weight positive examples (closure=True) by the ratio of negatives to positives. This effectively penalizes missing a true closure more than falsely predicting one — appropriate for the operational context (it's worse to not prepare for a closure than to over-prepare).

The evaluation metric was set to `aucpr` (area under the precision-recall curve) rather than `auc` (ROC-AUC). For imbalanced classification, precision-recall curves are more informative because they focus on the minority class performance rather than being diluted by the easy true-negatives.

### Cross-Validation Strategy for Task B

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

`StratifiedKFold` is used instead of plain `KFold` because class imbalance means random splits might have all positive examples in one fold by chance. Stratified splitting ensures each fold has approximately the same class ratio as the full dataset.

### Model Comparison (Task B)

Three models were compared via OOF evaluation:

**XGBoost** with Optuna-tuned hyperparameters (60 trials): Best AUC
**Random Forest** with `class_weight='balanced'`: Second best
**Logistic Regression** (Pipeline with StandardScaler): Baseline

XGBoost was selected as the production model.

### Final Model Saving

```python
joblib.dump(final_model_b,  os.path.join(MODEL_DIR, 'xgb_road_closure.pkl'))
joblib.dump(FEATURE_COLS_B, os.path.join(MODEL_DIR, 'feature_cols_B.pkl'))
joblib.dump(best_params,    os.path.join(MODEL_DIR, 'best_params_B.pkl'))
```

---

## Feature Engineering Deep Dive

### Complete Final Feature Set (42 Features)

| Feature | Type | Rationale |
|---|---|---|
| `hour` | Raw temporal | Hour of day (0–23) |
| `day_of_week` | Raw temporal | Monday=0 to Sunday=6 |
| `month` | Raw temporal | 1–12 |
| `week_of_year` | Raw temporal | 1–52, captures seasonal events |
| `quarter` | Raw temporal | 1–4 |
| `is_weekend` | Binary temporal | Higher baseline congestion on weekends |
| `is_peak` | Binary temporal | Combined AM+PM rush hour flag |
| `is_peak_am` | Binary temporal | 7–9 AM specific |
| `is_peak_pm` | Binary temporal | 5–7 PM specific |
| `is_night` | Binary temporal | 10 PM–2 AM: faster clearance, less congestion |
| `is_monday` | Binary temporal | Monday-specific pattern |
| `is_friday` | Binary temporal | Worst PM peak of the week |
| `hour_sin`, `hour_cos` | Cyclical | Circular time encoding (removes hour=23/0 discontinuity) |
| `dow_sin`, `dow_cos` | Cyclical | Circular day encoding |
| `month_sin`, `month_cos` | Cyclical | Circular month encoding |
| `latitude`, `longitude` | Geo | Raw coordinates |
| `lat_bin`, `lon_bin` | Geo | 8×8 geographic grid cell |
| `has_end_location` | Geo | Incident spans two points |
| `dist_from_center` | Geo | Euclidean distance from Bengaluru centroid |
| `event_type_enc` | Categorical | planned vs. unplanned (label encoded) |
| `event_cause_enc` | Categorical | 12 cause categories (label encoded) |
| `veh_type_enc` | Categorical | 11 vehicle types (label encoded) |
| `requires_road_closure` | Binary | Operational flag (Task A only) |
| `priority_enc` | Ordinal | Operator-assigned priority 0–3 |
| `zone_tenc` | Target encoded | Mean duration for this zone |
| `zone_count` | Count | Number of historical incidents in this zone |
| `junction_tenc` | Target encoded | Mean duration for this junction |
| `junction_count` | Count | Historical incidents at this junction |
| `corridor_tenc` | Target encoded | Mean duration for this corridor |
| `corridor_count` | Count | Historical incidents on this corridor |
| `event_cause_tenc` | Target encoded | Mean duration for this cause type |
| `event_cause_count` | Count | Historical incidents of this cause |
| `cause_x_peak` | Interaction | cause_enc × is_peak |
| `cause_x_weekend` | Interaction | cause_enc × is_weekend |
| `cause_x_closure` | Interaction | cause_enc × requires_road_closure (Task A) |
| `zone_x_peak` | Interaction | zone_tenc × is_peak |
| `closure_x_end` | Interaction | requires_road_closure × has_end_location (Task A) |

Task B uses a subset (38 features) with the 4 leaky features removed.

---

## Model Architecture & Selection

### Task A: XGBoost Regressor on Log-Duration

**Why XGBoost over LightGBM:**
Both models performed similarly in experiments. XGBoost was chosen because it's more battle-tested for tabular data with complex missing value patterns, and the serialized `.pkl` artifact is more portable.

**Why log transformation:**
Raw duration is right-skewed. `log1p` compresses the long tail, making the gradient updates more balanced across short and long incidents. At inference, `expm1()` recovers the original scale with a lower bound of 5 minutes applied (`max(5.0, round(duration, 1))`).

**Severity derivation (from duration, not a separate model):**
```
< 30 mins  → Low
30–90 mins → Medium
90–240 mins→ High
≥ 240 mins → Critical
```

**Confidence scoring:**
```python
boundaries = [30, 90, 240]
min_dist   = min(abs(duration - b) for b in boundaries)
confidence = "High" if min_dist > 20 else "Medium"
```
Predictions far from any severity boundary threshold are marked "High" confidence. Predictions near a boundary (e.g., 85 mins is close to the Low/Medium boundary at 90) are marked "Medium" confidence.

### Task B: XGBoost Classifier (Binary)

Uses `predict_proba()[:, 1]` clipped to `[0, 1]` for a smooth probability output. The probability is converted to a qualitative label:
- `> 0.7` → "Very likely"
- `0.4–0.7` → "Likely"
- `0.2–0.4` → "Possible"
- `< 0.2` → "Unlikely"

---

## Backend: FastAPI Service

The backend (`main.py`) is a FastAPI application that:

### Startup: Model Loading

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
```

All 6 model artifacts are loaded into the `MODELS` dict at startup using `joblib.load()`. A `/health` endpoint reports which models are loaded and whether the service is fully operational.

### Feature Construction at Inference (`build_feature_row`)

This function replicates the entire training pipeline in reverse — given a `PredictionRequest` (JSON input), it produces the same 42-dimensional feature vector the model was trained on. Every encoding decision made during training is mirrored here:

- `safe_le()`: Label encoder with fallback to first known class for unseen categories
- `tenc_val()`: Target encoding lookup with global mean fallback
- `PRIORITY_MAP`: Same ordinal encoding as training
- Cyclical features: Same sin/cos formulas
- Geographic binning: Same `lat_min/lat_max` bounds as training (12.80–13.27°N, 77.31–77.77°E for Bengaluru)
- Interaction features: Same product formulas

### Prediction Flow

```python
# Task A: Duration
log_pred = float(MODELS["xgb_clean"].predict(X_A)[0])
duration = float(np.expm1(log_pred))
duration = max(5.0, round(duration, 1))

# Severity from duration
if duration < 30:   sev_class = 0  # Low
elif duration < 90: sev_class = 1  # Medium
elif duration < 240:sev_class = 2  # High
else:               sev_class = 3  # Critical

# Task B: Road Closure Probability
closure_prob = float(MODELS["xgb_road_closure"].predict_proba(X_B)[0][1])
closure_prob = round(float(np.clip(closure_prob, 0.0, 1.0)), 3)
```

### Resource Recommendation Logic

Resource recommendations are rule-based on top of the ML output:

```python
manpower_map  = {"Low": (4,6), "Medium": (8,12), "High": (14,18), "Critical": (20,28)}
barricade_map = {"Low": 2,     "Medium": 4,       "High": 8,       "Critical": 14}
```

Diversions are activated if `closure_prob > 0.4` OR severity is High/Critical. Diversion routes are pre-defined per zone:
- Central Zone → Outer Ring Road via Hebbal, Tumkur Road diversion
- North Zone → Bellary Road alternate, NH44 bypass
- South Zone → Kanakapura Road bypass, Bannerghatta Road alternate
- East Zone → Old Madras Road alternate, Whitefield bypass
- West Zone → Magadi Road bypass, Mysore Road alternate

### Similar Events Lookup

A curated bank of 13 historical events covers all major cause categories. The lookup prioritizes:
1. Events matching both cause AND zone prefix
2. Events matching cause only
3. Any remaining events

This gives operators realistic benchmark cases to calibrate their expectations.

### Feature Importance Endpoint

```python
def get_feature_importances() -> List[FeatureImportance]:
    scores = model.feature_importances_
    pairs  = sorted(zip(feat_A, scores), key=lambda x: x[1], reverse=True)[:10]
    total  = sum(s for _, s in pairs) or 1.0
    return [FeatureImportance(feature=f, importance=round(float(s/total), 4)) ...]
```

XGBoost's `feature_importances_` attribute gives the average gain per feature across all trees. Normalizing to sum to 1.0 and taking the top 10 provides an interpretable explainability layer for the prediction.

---

## Frontend: React Dashboard

### Design System

The UI uses a GitHub Dark theme (`#0d1117` background, `#161b22` card surfaces, `#30363d` borders) with a monospace accent (`JetBrains Mono`) for metric values. The color palette maps to severity levels:
- Green `#3fb950` → Low / positive indicators
- Yellow `#d29922` → Medium / warnings
- Orange `#f78166` → High / alerts
- Red `#f85149` → Critical / danger

### Layout

**Two-panel layout:**
- Left panel (380px fixed): Input form with four sections (Classification, Location, Time Context, Incident Flags) and a submit button
- Right panel (flex): Results dashboard with four tabs

**Left Panel — Input Form:**
- Event type and vehicle type (side by side dropdowns)
- Event cause (full-width dropdown)
- Latitude/Longitude (side by side number inputs, defaulting to Bengaluru centroid)
- Zone and Junction selection
- Hour input + quick preset buttons (Morning peak, Afternoon, Evening peak, Night)
- Day of week selector
- Checkbox flags for road closure and end location

**Right Panel — Four Tabs:**

1. **Prediction Tab:** Hero metrics grid (duration, severity, road closure, manpower) + a road closure probability bar + diversion routes if needed

2. **Resources Tab:** Detailed resource cards + diversion route list + deployment checklist (5-step actionable plan)

3. **Explainability Tab:** Feature importance bar chart (`ImportanceChart` component) + input/output breakdown grid

4. **Similar Events Tab:** `SimilarEvents` component showing past incidents with matching cause/zone, duration, and severity tags

### State Management

All state is local React state using `useState`. The form state initializes with Bengaluru defaults:
```javascript
const DEFAULT_FORM = {
    event_type: 'unplanned',
    event_cause: 'vehicle_breakdown',
    latitude: '12.9716',
    longitude: '77.5946',
    hour: new Date().getHours(),
    day_of_week: ...,
    zone: 'Central Zone 1',
    ...
}
```

### Options Fetching

```javascript
useEffect(() => {
    fetchOptions()
      .then(setOptions)
      .catch(() => setOptions(HARDCODED_FALLBACK))
}, [])
```

On mount, the frontend fetches `/options` from the backend to populate dropdowns. If the backend is unavailable, a hardcoded fallback ensures the form still renders.

---

## How to Run

### Prerequisites

- Python 3.10+
- Node.js 18+
- A trained `models_v3/` directory (generated by running the notebooks in order)

### 1. Clone and Set Up

```bash
git clone https://github.com/spyke7/evensaathi.git
cd eventsaathi
```

### 2. Generate Model Artifacts (Run Notebooks)

Place your `dataset.csv` in the same directory as the notebooks and run them in order:

1. `fg2_clean1.ipynb` -> generates `models_v3/xgb_clean.pkl`, `feature_cols_clean.pkl`, `global_tenc_clean.pkl`, `label_encoders_clean.pkl`
2. `fg2_clean2.ipynb` -> loads the encoders from step 1 and generates `models_v3/xgb_road_closure.pkl`, `feature_cols_B.pkl`

### 3. Run the Backend

```bash
cd backend
pip install -r requirements.txt
(Move the models_v3 models to backend/models folder)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Verify the backend is healthy:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": true, "loaded_keys": [...]}
```

### 4. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

The development server starts at `http://localhost:5173`.

**Update the API base URL** in `src/api.js` if your backend runs on a different port:
```javascript
const BASE_URL = 'http://localhost:8000'
```

---

## Running the Notebooks

### Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna joblib matplotlib jupyter
```

### Execution Order

Run notebooks in this exact order:

```
1. fg2_exp3_2.ipynb           (optional — exploratory, takes ~15-30 mins due to Optuna)
2. fg2_exp3_2-Copy1.ipynb     (optional — documented rerun of above)
3. fg2_clean1.ipynb           (REQUIRED — trains Task A model, saves encoders)
4. fg2_clean2.ipynb           (REQUIRED — trains Task B model, loads Task A encoders)
```

**Important:** `fg2_clean2.ipynb` loads encoder artifacts from `models_v3/` saved by `fg2_clean1.ipynb`. Run them in sequence.

### Expected Outputs

After running `fg2_clean1.ipynb`:
- `models_v3/xgb_clean.pkl`
- `models_v3/feature_cols_clean.pkl`
- `models_v3/global_tenc_clean.pkl`
- `models_v3/label_encoders_clean.pkl`
- Console output showing OOF MAE, RMSE, R², and sanity check predictions

After running `fg2_clean2.ipynb`:
- `models_v3/xgb_road_closure.pkl`
- `models_v3/feature_cols_B.pkl`
- Console output showing OOF AUC, F1, and sanity check closure probabilities

---

## API Reference

### `POST /predict`

**Request Body:**
```json
{
  "event_type": "unplanned",
  "event_cause": "tree_fall",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "hour": 8,
  "day_of_week": 1,
  "zone": "North Zone 1",
  "junction": "Unknown",
  "veh_type": "Unknown",
  "requires_road_closure": true,
  "has_end_location": false,
  "corridor": "",
  "month": 6,
  "priority": "high"
}
```

**Response:**
```json
{
  "duration_mins": 127.4,
  "duration_label": "2h 7m",
  "severity": "High",
  "severity_score": 72,
  "road_closure_prob": 0.781,
  "road_closure_label": "Very likely",
  "resources": {
    "manpower_min": 14,
    "manpower_max": 18,
    "barricades": 8,
    "diversion_needed": true,
    "diversion_zones": ["Bellary Road alternate", "NH44 bypass"]
  },
  "similar_events": [...],
  "feature_importances": [...],
  "model_used": "XGBoost",
  "confidence": "High"
}
```

### `GET /options`

Returns available dropdown values for `event_types`, `event_causes`, `veh_types`, `zones`, `junctions`, `priorities`.

### `GET /health`

Returns service status and which model keys are loaded.

---

## Key Technical Decisions Summary

| Decision | What | Why |
|---|---|---|
| Log-transform target | `log1p(duration_mins)` | Right-skewed distribution; log scale balances gradient updates across incident durations |
| OOF cross-validation | 5-fold KFold (Task A), StratifiedKFold (Task B) | Small dataset; single split gives noisy estimates; OOF uses all data for both train and eval |
| Target encoding for locations | Zone, junction, corridor, cause | High cardinality (229 junctions); label encoding loses ordering; target encoding preserves operational meaning |
| Cyclical time features | sin/cos for hour, dow, month | Removes artificial discontinuity between hour 23 and hour 0 |
| Separate Task B encoder reuse | Load Task A's `le_encoders` and `global_tenc` | One feature row used for both models at inference; encodings must be identical |
| Leakage prevention for Task B | Remove `requires_road_closure`, `has_end_location`, derived interactions | These are correlated with the target variable; using them would give unrealistically high test AUC |
| `scale_pos_weight` for imbalanced classifier | `neg/pos` ratio | Road closures are minority class; standard training would predict "no closure" for everything |
| Interaction features | `cause × peak`, `zone × peak`, etc. | Small dataset limits depth at which tree models learn interactions; explicit products help |
| Single XGBoost vs. stacking | XGBoost only | Stacking improved MAE by ~2% but required 3 models at inference; operational cost outweighed marginal gain |