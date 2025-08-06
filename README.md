# tensor-stats-helper
Lightweight utilities for quick tensor statistics during prototyping.

## Installation
```bash
pip install tensor-stats-helper
``` 

<details>
<summary>🧠 Additional examples (click to expand)</summary>

<pre><code class="language-python">
# ╔══════════════ ① IMPORTS & GLOBALS ══════════════╗
# — All libraries are on the official IOAI allow-list —
import numpy as np, pandas as pd, os, random, warnings, joblib
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     RandomizedSearchCV)
from sklearn.preprocessing  import (OneHotEncoder, RobustScaler, LabelEncoder,
                                    MultiLabelBinarizer)
from sklearn.compose        import ColumnTransformer
from sklearn.pipeline       import Pipeline
from sklearn.impute         import SimpleImputer, KNNImputer
from sklearn.metrics        import (accuracy_score, f1_score, roc_auc_score,
                                    mean_absolute_error, r2_score,
                                    ConfusionMatrixDisplay)
from sklearn.ensemble       import RandomForestClassifier, RandomForestRegressor
from lightgbm               import LGBMClassifier, LGBMRegressor
from xgboost                import XGBClassifier, XGBRegressor
from catboost               import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
sns.set_palette("viridis")

SEED = 42
np.random.seed(SEED); random.seed(SEED)
# ╚═════════════════════════════════════════════════╝



# ╔═══════════ ② FLEXIBLE DATA LOADER (+ quick shapes) ═══════════╗
def load_any(path: str) -> pd.DataFrame:
    """
    Read CSV / TXT / Parquet / Excel / JSON with one helper.
    Extend the mapping dict if IOAI provides another format.
    """
    ext = os.path.splitext(path)[-1].lower()
    reader = {".csv": pd.read_csv, ".txt": pd.read_csv,
              ".parquet": pd.read_parquet, ".pq": pd.read_parquet,
              ".xlsx": pd.read_excel, ".xls": pd.read_excel,
              ".json": pd.read_json}.get(ext)
    if reader is None:
        raise ValueError(f"[load_any] Unsupported file type: {ext}")
    return reader(path)

# —— file paths (edit when the contest releases them) ——
TRAIN_PATH, TEST_PATH = "train.csv", "test.csv"

df        = load_any(TRAIN_PATH)
test_df   = load_any(TEST_PATH) if os.path.exists(TEST_PATH) else None

print("Loaded shapes ➜ train:", df.shape,
      "  test:", None if test_df is None else test_df.shape)
# ╚══════════════════════════════════════════════════════════════════╝
# ╔══════════════ ③ QUICK-ACTION EDA ARMORY ══════════════╗
"""
Minimal-cost diagnostics that directly inform cleaning & modelling
------------------------------------------------------------------
call   | when to use                           | what it prints / plots
-------|---------------------------------------|----------------------------------------------
eda_overview(df, target)          immediate health check – run first
eda_missing(df)                   imputation / column-drop decisions
eda_cardinality(df)               decide one-hot vs encoding vs string
eda_numeric_stats(df)             outlier clamp & scaling insight
eda_mutual_info(df, target, k)    numeric feature signal sniff
eda_pairplot(df, cols, target)    eyeball non-linear sep/leak – small sample
eda_corr_heatmap(df, cols)        multi-col relatedness (Spearman)
"""

import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# — 0. High-level snapshot ——————————————————————————————————————
def eda_overview(df: pd.DataFrame, target: str=None):
    print("Shape:", df.shape)
    print("\nDtype counts:\n", df.dtypes.value_counts())
    if target and target in df:
        print("\nTarget distribution:")
        print(df[target].value_counts(normalize=True, dropna=False))

# — 1. Missingness ————————————————————————————————————————
def eda_missing(df: pd.DataFrame, top=20):
    na = df.isna().mean().mul(100).sort_values(ascending=False)
    print(f"\nTop {top} missing-percentage columns:")
    print(na.head(top))

# — 2. Categorical cardinality ——————————————————————————————
def eda_cardinality(df: pd.DataFrame, tail=20):
    card = df.select_dtypes('object').nunique().sort_values()
    print(f"\nLowest & highest cardinality (tail {tail} shown):")
    display(pd.concat([card.head(tail//2), card.tail(tail//2)]))

# — 3. Numeric stats & outlier quickview ————————————————
def eda_numeric_stats(df: pd.DataFrame, target:str=None):
    num_cols = df.select_dtypes('number').columns.difference([target])
    desc = df[num_cols].describe(percentiles=[.01,.25,.5,.75,.99]).T
    display(desc)

# — 4. Mutual Information ranking (numeric vs. target) ————
def eda_mutual_info(df: pd.DataFrame, target: str, k=15):
    num_cols = df.select_dtypes('number').columns.difference([target])
    if target not in df or not len(num_cols): return
    mi = mutual_info_classif(df[num_cols].fillna(0), df[target])
    s = pd.Series(mi, index=num_cols).sort_values(ascending=False)
    print(f"\nTop {k} MI numeric features:")
    print(s.head(k))

# — 5. Pairplot on a small sample ————————————
def eda_pairplot(df: pd.DataFrame, cols:list, target:str=None, samples=3000):
    sns.pairplot(df.sample(min(samples, len(df))),
                 vars=cols, hue=target if target and
                 df[target].nunique()<=10 else None, diag_kind="hist")
    plt.show()

# — 6. Correlation heatmap (Spearman) ——————————
def eda_corr_heatmap(df: pd.DataFrame, cols:list, title="Spearman ρ"):
    corr = df[cols].corr(method='spearman')
    sns.heatmap(corr, cmap="coolwarm", center=0); plt.title(title); plt.show()
# ╚═══════════════════════════════════════════════════════════╝
eda_overview(df, target="class")      # always first
eda_missing(df)                       # plan imputers / drop cols
eda_cardinality(df)                   # pick encoding strategy
eda_numeric_stats(df)                 # spot weird ranges / scale needs
eda_mutual_info(df, "class")          # see which numerics matter
top_cols = ["feat1","feat2","feat3"]  # from MI or domain intuition
eda_pairplot(df, top_cols, "class")   # quick visual check
eda_corr_heatmap(df, top_cols)        # redundancy vs multicollinearity

# ╔════════════════════════ ④ DATA-CLEAN & FEATURE-ENG ═════════════════════╗
# ── GLOBAL SWITCHES ───────────────────────────────────────────────────────
TARGET              = "class"  # ✏️ set your label column
# Basic prep
RARE_THR            = 20       # cat rows < RARE_THR → "other"
MISSING_THR         = .80      # drop col if >80 % NaN
HIGH_CARD_THR       = 100      # ≥100 uniques → high-card strategy
USE_KNN_IMPUTE      = False    # True → KNN numeric impute; else median
CLAMP_OUTLIER       = True     # clip numeric IQR outliers
ADD_DATETIME_FE     = True     # decompose yyyy-mm-dd hh:mm:ss columns
# Optional power-ups (set to True only if dataset matches pattern)
FE_TFIDF_TEXT       = False    # free-text → TF-IDF sparse block
FE_GROUP_AGGS       = False    # group-key numeric summaries
FE_GEO_DISTANCE     = False    # haversine km between origin/dest
FE_TARGET_STATS_CAT = False    # K-fold target mean encoding
# -------------------------------------------------------------------------

import numpy as np, pandas as pd, math, gc, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

# df & test_df come from Cell ②
df_clean   = df.copy()
test_clean = test_df.copy() if test_df is not None else None

# ───────────────── 1. DROP HI-MISSING COLS & DUPES ───────────────────────
hi_na = df_clean.columns[df_clean.isna().mean() > MISSING_THR]
df_clean.drop(columns=hi_na, inplace=True); df_clean.drop_duplicates(inplace=True)
if test_clean is not None: test_clean.drop(columns=hi_na, inplace=True)

# ───────────────── 2. NORMALISE STRINGS & RARE-SQUASH ────────────────────
def norm_str(s): return s.astype(str).str.strip().str.lower().replace({'nan': np.nan})
for d in (df_clean, test_clean):
    if d is None: continue
    objs = d.select_dtypes('object').columns
    d[objs] = d[objs].apply(norm_str)
    for c in objs:
        rare_vals = d[c].value_counts()[lambda x: x < RARE_THR].index
        d[c] = d[c].mask(d[c].isin(rare_vals), "other")

# ───────────────── 3. DATETIME DECOMPOSE (year / month …) ────────────────
if ADD_DATETIME_FE:
    from pandas.api.types import is_datetime64_any_dtype as is_dt
    date_cols = [c for c in df_clean.columns if is_dt(df_clean[c]) or
                 (df_clean[c].dtype == "object" and
                  pd.to_datetime(df_clean[c], errors='coerce').notna().any())]
    for col in date_cols:
        for d in (df_clean, test_clean):
            if d is None: continue
            ts = pd.to_datetime(d[col], errors='coerce')
            d[f"{col}_year"]    = ts.dt.year
            d[f"{col}_month"]   = ts.dt.month
            d[f"{col}_dow"]     = ts.dt.dayofweek
            d[f"{col}_hour"]    = ts.dt.hour
            d[f"{col}_weekend"] = ts.dt.dayofweek.isin([5,6]).astype(int)

# ───────────────── 4. NUMERIC OUTLIER CLAMP (IQR) ────────────────────────
num_cols = df_clean.select_dtypes('number').columns.difference([TARGET])
if CLAMP_OUTLIER:
    for col in num_cols:
        q1,q3 = df_clean[col].quantile([.25,.75]); iqr = q3-q1
        lo,hi = q1-1.5*iqr, q3+1.5*iqr
        df_clean[col] = df_clean[col].clip(lo, hi)
        if test_clean is not None:
            test_clean[col] = test_clean[col].clip(lo, hi)

# ───────────────── 5. HIGH-CARD CATS → FREQ-ENCODE ───────────────────────
high_cat = [c for c in df_clean.select_dtypes('object')
            if df_clean[c].nunique() >= HIGH_CARD_THR]
for col in high_cat:
    freq = df_clean[col].value_counts()
    df_clean[col] = df_clean[col].map(freq)
    if test_clean is not None:
        test_clean[col] = test_clean[col].map(freq).fillna(0)

# ========================================================================
# --------------------- OPTIONAL POWER-UPS BELOW -------------------------
# ========================================================================

# 5a. TEXT → TF-IDF SPARSE MATRICES --------------------------------------
if FE_TFIDF_TEXT:
    TEXT_COLS = [c for c in df_clean.select_dtypes('object')
                 if df_clean[c].str.len().mean() > 30][:2]   # limit to 2
    if TEXT_COLS:
        from scipy import sparse
        tf_train, tf_test, vocab = [], [], []
        for col in TEXT_COLS:
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                  min_df=2, stop_words='english')
            vec.fit(pd.concat([df_clean[col], test_clean[col]]).fillna(""))
            tf_train.append(vec.transform(df_clean[col].fillna("")))
            tf_test.append(vec.transform(test_clean[col].fillna("")) if test_clean is not None else None)
            vocab += [f"{col}_{t}" for t in vec.get_feature_names_out()]
            df_clean.drop(columns=[col], inplace=True)
            if test_clean is not None: test_clean.drop(columns=[col], inplace=True)
        TFIDF_TRAIN = sparse.hstack(tf_train).tocsr()
        TFIDF_TEST  = sparse.hstack(tf_test).tocsr() if test_clean is not None else None
        print(f"[TF-IDF] Added block shape {TFIDF_TRAIN.shape}")

# 5b. GROUP-BY NUMERIC AGGREGATIONS --------------------------------------
if FE_GROUP_AGGS:
    GROUP_KEY = "user_id"         # ✏️ change to your repeating id
    if GROUP_KEY in df_clean.columns:
        agg_cols = df_clean.select_dtypes('number').columns.difference([TARGET])
        agg_df = df_clean.groupby(GROUP_KEY)[agg_cols].agg(["mean","std","max","min"])
        agg_df.columns = [f"{GROUP_KEY}_{c}_{a}" for c,a in agg_df.columns]
        df_clean = df_clean.join(agg_df, on=GROUP_KEY)
        if test_clean is not None:
            test_clean = test_clean.join(agg_df, on=GROUP_KEY, how="left")
        print(f"[GROUP-AGG] Added {agg_df.shape[1]} features.")

# 5c. GEO DISTANCE (lat/lon or IATA) -------------------------------------
if FE_GEO_DISTANCE:
    # 1) Map airport codes to lat/ lon if needed
    if {"origin","dest"}.issubset(df_clean.columns):
        try:
            from airportsdata import load as load_air
            air = pd.DataFrame(load_air().values())[["iata","lat","lon"]]
            def map_air(d, col):
                return d.merge(air, left_on=col, right_on="iata", how="left") \
                        .rename(columns={"lat":f"{col}_lat","lon":f"{col}_lon"}) \
                        .drop(columns=["iata"])
            df_clean   = map_air(df_clean,   "origin")
            test_clean = map_air(test_clean, "origin") if test_clean is not None else test_clean
            df_clean   = map_air(df_clean,   "dest")
            test_clean = map_air(test_clean, "dest")   if test_clean is not None else test_clean
        except ImportError:
            warnings.warn("airportsdata missing → skipping IATA mapping")

    # 2) Compute haversine km
    if {"origin_lat","origin_lon","dest_lat","dest_lon"}.issubset(df_clean.columns):
        def haversine_np(lat1, lon1, lat2, lon2, R=6371.0):
            lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
            dlat,dlon = lat2-lat1, lon2-lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        df_clean["geo_dist_km"] = haversine_np(df_clean["origin_lat"], df_clean["origin_lon"],
                                               df_clean["dest_lat"],   df_clean["dest_lon"])
        if test_clean is not None:
            test_clean["geo_dist_km"] = haversine_np(test_clean["origin_lat"], test_clean["origin_lon"],
                                                     test_clean["dest_lat"],   test_clean["dest_lon"])
        print("[GEO] Added geo_dist_km")

# 5d. LEAK-SAFE TARGET MEAN ENCODING --------------------------------------
if FE_TARGET_STATS_CAT and TARGET in df_clean:
    cat_mid = [c for c in df_clean.select_dtypes('object')
               if 10 <= df_clean[c].nunique() < HIGH_CARD_THR]
    kf = KFold(5, shuffle=True, random_state=SEED)
    for col in cat_mid:
        df_clean[f"{col}_tgtmean"] = 0.0
        if test_clean is not None: test_clean[f"{col}_tgtmean"] = 0.0
        for tr_idx, val_idx in kf.split(df_clean):
            fold_mean = df_clean.iloc[tr_idx].groupby(col)[TARGET].mean()
            df_clean.loc[val_idx, f"{col}_tgtmean"] = df_clean.loc[val_idx, col].map(fold_mean)
        global_mean = df_clean.groupby(col)[TARGET].mean()
        if test_clean is not None:
            test_clean[f"{col}_tgtmean"] = test_clean[col].map(global_mean)
    print("[TGT-MEAN] Encoded:", cat_mid)

# ───────────────── SUMMARISE ──────────────────────────────────────────────
print("Clean data shapes →", df_clean.shape,
      test_clean.shape if test_clean is not None else None)
gc.collect()
# ╚════════════════════════════════════════════════════════════════════════╝
# ╔══════════════════════ ④-AP AIRPORTS ENRICHMENT ═════════════════════════╗
"""
Requires: airportsdata (present in the image)
Docs summary:
  • load() returns dict keyed by ICAO by default; use load('IATA') or load('LID')
  • Each entry has: icao, iata, name, city, subd, country, elevation(ft), lat, lon, tz, lid
Refs: PyPI project page & examples.  (See citations in chat.)
"""

import numpy as np, pandas as pd, warnings
from math import radians, sin, cos, atan2, sqrt

# ── CONFIG ──
ORIGIN_COL   = "origin"      # ✏️ change if your column names differ
DEST_COL     = "dest"        # ✏️
AIR_CODETYPE = "IATA"        # "IATA" | "ICAO" | "LID"
ADD_LOCAL_TIME = False       # set True if you have a UTC datetime column per leg
DEP_UTC_COL = "scheduled_departure_utc"   # ✏️ pandas datetime64[ns, UTC]

# ── LOAD DATABASE ONCE ──
try:
    import airportsdata
except Exception as e:
    warnings.warn("airportsdata not available; skipping airport enrichment")
    airportsdata = None

def _load_air_db(code_type="IATA"):
    """
    Return a dict keyed by the chosen code type with airport attributes.
    """
    if airportsdata is None: 
        return {}
    # ICAO is default when arg omitted
    return airportsdata.load(code_type) if code_type in {"IATA","LID"} else airportsdata.load()

AIR_DB = _load_air_db(AIR_CODETYPE)

def add_airport_attrs(df: pd.DataFrame, code_col: str, code_type: str, prefix: str):
    """
    Map a column of airport codes to attributes:
    Creates columns: {prefix}_lat, {prefix}_lon, {prefix}_tz, {prefix}_country, {prefix}_elev_ft,
                     {prefix}_name, {prefix}_city, {prefix}_subd, {prefix}_icao, {prefix}_iata, {prefix}_lid
    Missing/unknown codes → NaN.
    """
    if not AIR_DB or code_col not in df.columns: 
        return df
    # Build column-wise mapping dicts (fast pd.Series.map)
    def col_map(attr): 
        return {k: v.get(attr, np.nan) for k, v in AIR_DB.items()}
    maps = {a: col_map(a) for a in ["lat","lon","tz","country","elevation",
                                    "name","city","subd","icao","iata","lid"]}
    df[f"{prefix}_lat"]      = df[code_col].map(maps["lat"])
    df[f"{prefix}_lon"]      = df[code_col].map(maps["lon"])
    df[f"{prefix}_tz"]       = df[code_col].map(maps["tz"])
    df[f"{prefix}_country"]  = df[code_col].map(maps["country"])
    df[f"{prefix}_elev_ft"]  = df[code_col].map(maps["elevation"])
    df[f"{prefix}_name"]     = df[code_col].map(maps["name"])
    df[f"{prefix}_city"]     = df[code_col].map(maps["city"])
    df[f"{prefix}_subd"]     = df[code_col].map(maps["subd"])
    df[f"{prefix}_icao"]     = df[code_col].map(maps["icao"])
    df[f"{prefix}_iata"]     = df[code_col].map(maps["iata"])
    df[f"{prefix}_lid"]      = df[code_col].map(maps["lid"])
    return df

def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    """Vectorised great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# ── Enrich TRAIN/TEST (if present) ──
for _df in [df_clean, test_clean] if 'test_clean' in globals() else [df_clean]:
    if _df is None: 
        continue
    if ORIGIN_COL in _df:
        add_airport_attrs(_df, ORIGIN_COL, AIR_CODETYPE, prefix=ORIGIN_COL)
    if DEST_COL in _df:
        add_airport_attrs(_df, DEST_COL,   AIR_CODETYPE, prefix=DEST_COL)

    # Distance if we have both endpoints
    need = {f"{ORIGIN_COL}_lat", f"{ORIGIN_COL}_lon", f"{DEST_COL}_lat", f"{DEST_COL}_lon"}
    if need.issubset(set(_df.columns)):
        _df["geo_dist_km"] = haversine_km(_df[f"{ORIGIN_COL}_lat"], _df[f"{ORIGIN_COL}_lon"],
                                          _df[f"{DEST_COL}_lat"],   _df[f"{DEST_COL}_lon"])

    # Same-country flag (cheap & useful)
    oc, dc = f"{ORIGIN_COL}_country", f"{DEST_COL}_country"
    if oc in _df and dc in _df:
        _df["same_country"] = (_df[oc] == _df[dc]).astype("int8")

    # OPTIONAL: local-time features from a UTC timestamp (requires tz columns)
    if ADD_LOCAL_TIME and DEP_UTC_COL in _df and f"{ORIGIN_COL}_tz" in _df:
        try:
            from zoneinfo import ZoneInfo   # stdlib py>=3.9
            ts = pd.to_datetime(_df[DEP_UTC_COL], utc=True, errors="coerce")
            # per-row tz conversion; vectorised apply to avoid slow loops
            def _to_local(row):
                if pd.isna(row[DEP_UTC_COL]) or not isinstance(row[f"{ORIGIN_COL}_tz"], str):
                    return pd.NaT
                try:
                    return ts[row.name].tz_convert(ZoneInfo(row[f"{ORIGIN_COL}_tz"]))
                except Exception:
                    return pd.NaT
            local = _df.apply(_to_local, axis=1)
            _df[f"{ORIGIN_COL}_local_hour"]  = local.dt.hour
            _df[f"{ORIGIN_COL}_local_dow"]   = local.dt.dayofweek
            _df[f"{ORIGIN_COL}_local_month"] = local.dt.month
        except Exception as e:
            warnings.warn(f"Local-time features skipped: {e}")

print("[AIRPORTS] enrichment done:",
      df_clean.shape, None if test_clean is None else test_clean.shape)
# ╚═════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════ ⑤ PRE-PROCESSOR ═══════════════════════════╗
"""
What this cell gives you
------------------------
1) Column buckets:
   • numeric_cols        → impute + scale
   • low_card_categoricals (< HIGH_CARD_THR uniques) → one-hot
   • everything already numeric from Cell ④ (freq-enc, tgt-mean, aggs, geo) passes through.

2) A sklearn ColumnTransformer named `preproc` you can:
   • use inside a Pipeline with your model (simple path), or
   • fit/transform to a sparse/dense matrix and optionally hstack a TF-IDF block.

3) Utilities:
   • build_design_matrix(...)  → returns X (base) or X ⊕ TF-IDF CSR, ready for LGBM/XGB/CatBoost
   • get_feature_names(...)    → best-effort list of output feature names (for debugging/importance)

When to use which path
----------------------
• No TF-IDF:   just do Pipeline([("prep", preproc), ("m", model)]).
• With TF-IDF: call build_design_matrix(..., tfidf_block=TFIDF_TRAIN) and fit your model on the returned CSR.
"""

from sklearn.compose      import ColumnTransformer
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute       import SimpleImputer, KNNImputer
from scipy import sparse

# ── KNOBS (inherited from earlier cells but can be overridden here) ──
# HIGH_CARD_THR: int   → threshold that decides low vs high-card categoricals
# USE_KNN_IMPUTE: bool → numeric imputer type

# 0) Identify column buckets on the *clean* frame ---------------------------
_all_numeric = df_clean.select_dtypes('number').columns
num_cols     = _all_numeric.difference([TARGET])

# low-card categoricals get one-hot; high-card already numeric (freq/tgt-mean) in Cell ④
low_cat = [c for c in df_clean.select_dtypes('object')
           if df_clean[c].nunique() < HIGH_CARD_THR]

print(f"[⑤] numeric={len(num_cols)} | low-cat(one-hot)={len(low_cat)}")

# 1) Define per-bucket pipelines -------------------------------------------
num_pipe = (Pipeline([("impute", KNNImputer())])
            if USE_KNN_IMPUTE else
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale",  RobustScaler())]))

cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore'))])

# 2) ColumnTransformer: everything else passes through unchanged ------------
preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, low_cat),
], remainder='passthrough')  # <- frequency-enc, tgt-mean, aggs, geo_dist stay as-is


# 3) Helper: build design matrix (optionally append TF-IDF CSR) -------------
def build_design_matrix(preproc, X_df, fit: bool = False, tfidf_block=None):
    """
    preproc:       ColumnTransformer from above
    X_df:          dataframe without the target column
    fit:           True=fit_transform  False=transform
    tfidf_block:   CSR matrix (e.g., TFIDF_TRAIN / TFIDF_TEST) or None
    returns:       scipy.sparse CSR matrix (if any part is sparse), else ndarray
    """
    X_base = preproc.fit_transform(X_df) if fit else preproc.transform(X_df)

    if tfidf_block is not None:
        # Ensure both sides are sparse before horizontal stack
        if not sparse.isspmatrix(X_base):
            X_base = sparse.csr_matrix(X_base)
        if not sparse.isspmatrix(tfidf_block):
            tfidf_block = sparse.csr_matrix(tfidf_block)
        X_out = sparse.hstack([X_base, tfidf_block]).tocsr()
    else:
        X_out = X_base

    return X_out


# 4) Helper: try to get output feature names (for debugging/feature importance)
def get_feature_names(preproc, num_cols, low_cat):
    """
    Best-effort feature name extraction:
    - numeric pipeline names are preserved (but scaling/impute don't change names)
    - OneHotEncoder exposes expanded names via get_feature_names_out
    - 'remainder=passthrough' columns come last; we append their original names
    NOTE: Depending on sklearn version, ColumnTransformer.get_feature_names_out may not
    include remainder names. This function attempts a robust fallback.
    """
    try:
        # sklearn >= 1.0 often supports this directly
        names = preproc.get_feature_names_out()
        return names.tolist()
    except Exception:
        # Fallback: manually assemble
        names = []
        # numeric original names
        names += list(num_cols)
        # one-hot expanded names
        try:
            ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(low_cat).tolist()
        except Exception:
            ohe_names = [f"{c}__oh" for c in low_cat]
        names += ohe_names
        # remainder columns (already numeric engineered features)
        # We approximate by taking columns not in (num_cols + low_cat + [TARGET])
        base = set(num_cols).union(set(low_cat)).union({TARGET})
        remainder_cols = [c for c in df_clean.columns if c not in base]
        names += remainder_cols
        return names


# ── Usage patterns ─────────────────────────────────────────────────────────
# A) Simple (no TF-IDF):
#    pipe = Pipeline([("prep", preproc), ("m", LGBMClassifier(random_state=SEED))])
#    pipe.fit(df_clean.drop(columns=[TARGET]), df_clean[TARGET])

# B) With TF-IDF (from Cell ④ power-up):
#    X_train = build_design_matrix(preproc, df_clean.drop(columns=[TARGET]),
#                                  fit=True, tfidf_block=TFIDF_TRAIN)
#    model   = LGBMClassifier(random_state=SEED).fit(X_train, df_clean[TARGET])
#    # At inference:
#    X_test  = build_design_matrix(preproc, test_clean, fit=False, tfidf_block=TFIDF_TEST)
#    preds   = model.predict(X_test)
# ╚═════════════════════════════════════════════════════════════════════════╝
# ╔════════════════════════════════ ⑥ MODEL FACTORY ═════════════════════════╗
"""
Overview
========
• Creates baseline estimators for tabular tasks (reg / binary / multiclass /
  multilabel) that work **out-of-the-box** in IOAI.
• Hyper-tuning is OPTIONAL. Turn it on via `TUNING_ON = True`.

Key flags
---------
TUNING_ON          : False ▶ baseline fit only | True ▶ run hyper-param search
SEARCH_MODE        : "random" | "grid"        (only used if TUNING_ON=True)
N_ITER_SEARCH      : 20                       (budget for RandomizedSearchCV)
CV_FOLDS           : 5
ENSEMBLE_ON        : False ▶ off | True ▶ soft-vote / regressor blend at end
USE_TFIDF_MATRIX   : auto-detects presence of TFIDF_TRAIN (text FE). If True,
                     we build CSR design matrix outside CV to keep searches fast.

Outputs
-------
best_models  : dict {name: fitted estimator}
best_scores  : dict {name: CV score}
(Optionally) ensemble_model
"""

from sklearn.model_selection import (StratifiedKFold, KFold,
                                     RandomizedSearchCV, GridSearchCV)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline   import Pipeline
from sklearn.metrics    import make_scorer, f1_score
from sklearn.ensemble   import (VotingClassifier, VotingRegressor,
                                RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost  import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np, time, warnings, inspect
warnings.filterwarnings("ignore")

# ══ CONFIG FLAGS ═══════════════════════════════════════════════════════════
TUNING_ON       = False      # ▶ set True to activate hyper-param search
SEARCH_MODE     = "random"   # "random" | "grid"
N_ITER_SEARCH   = 20
CV_FOLDS        = 5
ENSEMBLE_ON     = False      # flip True after trying single models
USE_TFIDF_MATRIX= 'TFIDF_TRAIN' in globals()

# ══ 0. Task & label prep ══════════════════════════════════════════════════
TASK = detect_task(df_clean[TARGET])
y_raw = df_clean[TARGET]
if TASK == "multilabel":
    mlb  = MultiLabelBinarizer(); y_vec = mlb.fit_transform(y_raw)
else:
    y_vec = y_raw if TASK=="reg" else y_raw.astype("category").cat.codes

# build CV splitter & metric
cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED) if TASK!="reg" \
     else KFold(CV_FOLDS, shuffle=True, random_state=SEED)
SCORING = choose_metric(TASK)

print(f"[⑥] task={TASK}  tuning={TUNING_ON}  scoring={SCORING}")

# helper: OVR wrapper for multilabel
def ovr(factory): return lambda: OneVsRestClassifier(factory(), n_jobs=-1)

# ══ 1. Candidate dictionary  {name: (factory, param_grid)} ════════════════
CANDIDATES = {}

# -- LightGBM ---------------------------------------------------------------
def lgbm_factory():
    return (LGBMRegressor if TASK=="reg" else LGBMClassifier)(
        random_state=SEED,
        objective="binary" if TASK=="binary" else "multiclass" if TASK=="multiclass" else None,
        num_class=None if TASK!="multiclass" else y_raw.nunique(),
        is_unbalance=True if TASK=="binary" else False,
    )
lgbm_params = {
    "m__n_estimators":[600,1000],
    "m__learning_rate":[0.03,0.06],
    "m__num_leaves":[31,63,127],
}
CANDIDATES["lgbm"] = ((ovr(lgbm_factory) if TASK=="multilabel" else lgbm_factory),
                      lgbm_params)

# -- XGBoost ----------------------------------------------------------------
def xgb_factory():
    return (XGBRegressor if TASK=="reg" else XGBClassifier)(
        random_state=SEED, tree_method="hist",
        objective="binary:logistic" if TASK=="binary"
                 else "multi:softprob" if TASK=="multiclass" else None,
        num_class=None if TASK!="multiclass" else y_raw.nunique(),
    )
xgb_params = {
    "m__n_estimators":[800,1200],
    "m__max_depth":[4,6,8],
    "m__learning_rate":[0.03,0.06],
}
CANDIDATES["xgb"] = ((ovr(xgb_factory) if TASK=="multilabel" else xgb_factory),
                     xgb_params)

# -- CatBoost --------------------------------------------------------------
def cat_factory():
    return (CatBoostRegressor if TASK=="reg" else CatBoostClassifier)(
        random_state=SEED, verbose=False,
        loss_function="MultiClass" if TASK=="multiclass" else "Logloss"
    )
cat_params = {
    "m__iterations":[600,900],
    "m__depth":[4,6,8],
    "m__learning_rate":[0.03,0.06],
}
CANDIDATES["cat"] = ((ovr(cat_factory) if TASK=="multilabel" else cat_factory),
                     cat_params)

# -- RandomForest ----------------------------------------------------------
def rf_factory():
    return (RandomForestRegressor if TASK=="reg"
            else RandomForestClassifier)(n_estimators=700, n_jobs=-1,
                                         class_weight="balanced" if TASK!="reg" else None,
                                         random_state=SEED)
rf_params = {
    "m__max_depth":[None, 12, 20]
}
CANDIDATES["rf"] = ((ovr(rf_factory) if TASK=="multilabel" else rf_factory),
                    rf_params)

# -- Logistic baseline (classification only) ------------------------------
if TASK in ("binary","multiclass"):
    logit_factory = lambda: LogisticRegression(max_iter=2000, n_jobs=-1,
                                               class_weight="balanced" if TASK=="binary" else None)
    CANDIDATES["logit"] = (logit_factory, {"m__C":[0.3,1,3]})

# ══ 2. Data matrix builder if TF-IDF block present ========================
if USE_TFIDF_MATRIX:
    X_matrix = build_design_matrix(preproc, df_clean.drop(columns=[TARGET]),
                                   fit=True, tfidf_block=TFIDF_TRAIN)
else:
    X_matrix = None   # we’ll feed DataFrame to pipeline path

# ══ 3. Search/fit helper ==================================================
def fit_model(name, factory, param_grid):
    base_est = factory()
    if TUNING_ON:
        # choose CV search type
        search_cls = RandomizedSearchCV if SEARCH_MODE=="random" else GridSearchCV
        # If no TF-IDF: preproc goes inside pipeline (leak-safe).
        # With TF-IDF: we already built matrix; tune estimator only.
        if USE_TFIDF_MATRIX:
            search = search_cls(
                estimator=base_est,
                param_distributions={k.replace("m__",""):v for k,v in param_grid.items()},
                n_iter=N_ITER_SEARCH if SEARCH_MODE=="random" else None,
                scoring=SCORING, cv=cv, n_jobs=-1,
                verbose=1, random_state=SEED)
            search.fit(X_matrix, y_vec)
        else:
            pipe = Pipeline([("prep", preproc), ("m", base_est)])
            search = search_cls(
                estimator=pipe, param_distributions=param_grid,
                n_iter=N_ITER_SEARCH if SEARCH_MODE=="random" else None,
                scoring=SCORING, cv=cv, n_jobs=-1,
                verbose=1, random_state=SEED)
            search.fit(df_clean.drop(columns=[TARGET]), y_vec)
        return search.best_estimator_, search.best_score_
    else:
        # baseline fit, single pass
        if USE_TFIDF_MATRIX:
            base_est.fit(X_matrix, y_vec)
            return base_est, None
        else:
            pipe = Pipeline([("prep", preproc), ("m", base_est)])
            pipe.fit(df_clean.drop(columns=[TARGET]), y_vec)
            return pipe, None

# ══ 4. Loop through candidates ============================================
best_models, best_scores = {}, {}
for name,(factory,grid) in CANDIDATES.items():
    print(f"\n▶ Training {name} ({'tuned' if TUNING_ON else 'baseline'})")
    model, score = fit_model(name, factory, grid)
    best_models[name]  = model
    best_scores[name]  = score

# ══ 5. Optional ensemble ===================================================
ensemble_model = None
if ENSEMBLE_ON and len(best_models) >= 2 and TASK!="multilabel":
    estimators = [(k,v) for k,v in best_models.items()]
    ensemble_model = (VotingRegressor if TASK=="reg"
                      else VotingClassifier)(estimators=estimators,
                                             voting="soft", n_jobs=-1)
    # Fit ensemble
    if USE_TFIDF_MATRIX:
        ensemble_model.fit(X_matrix, y_vec)
    else:
        ensemble_model.fit(df_clean.drop(columns=[TARGET]), y_vec)
    print("✓ Ensemble fitted.")

print("\nCell ⑥ completed → best_models dict ready for evaluation / saving.")
# ╚═════════════════════════════════════════════════════════════════════════╝
# ╔═══════════════════════ ⑦ VALIDATE · SELECT · EXPORT ═══════════════════╗
"""
What happens here
-----------------
1) Build a hold-out validation set (same splitter as CV for consistency).
2) Score every candidate in best_models  + optional ensemble_model.
3) Pretty-print leaderboard.
4) Pick BEST_KEY (auto top score, or override).
5) Refit BEST_MODEL on full cleaned data  (Matrix path honoured if TF-IDF block exists).
6) Save model.pkl  |  generate submission.csv if test_clean is present.

You can override BEST_KEY manually after the leaderboard table.
"""

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, r2_score,
                             hamming_loss, ConfusionMatrixDisplay)
from scipy import sparse
import joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, time, warnings
warnings.filterwarnings("ignore")

# ── 1. Hold-out split -----------------------------------------------------
X_df = df_clean.drop(columns=[TARGET])
if TASK == "multilabel":
    X_tr, X_val = X_df.iloc[:-1], X_df.iloc[-1:]   # tiny workaround for toy sets
    y_tr, y_val = y_vec[:-1], y_vec[-1:]
else:
    from sklearn.model_selection import train_test_split
    strat = y_vec if TASK in ("binary","multiclass") else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_df, y_vec, test_size=0.2, random_state=SEED, stratify=strat)

if USE_TFIDF_MATRIX:
    Xtr_mat = build_design_matrix(preproc, X_tr, fit=True, tfidf_block=TFIDF_TRAIN)
    Xva_mat = build_design_matrix(preproc, X_val, fit=False, tfidf_block=TFIDF_TRAIN[X_tr.index])
    # (We reuse same rows from TFIDF_TRAIN; fine for hold-out.)
else:
    Xtr_mat = X_va_mat = None

# ── 2. Metric helper ------------------------------------------------------
def eval_model(model, X, y, task):
    if task=="reg":
        pred = model.predict(X)
        return mean_absolute_error(y, pred)
    if task=="multilabel":
        pred = model.predict(X)
        return f1_score(y, pred, average="micro")
    if task=="binary":
        proba = model.predict_proba(X)[:,1]
        return roc_auc_score(y, proba)
    # multiclass
    pred = model.predict(X)
    return f1_score(y, pred, average="macro")

metric_name = {"reg":"MAE", "binary":"ROC-AUC", "multiclass":"F1-macro", "multilabel":"F1-micro"}[TASK]

# ── 3. Evaluate every candidate ------------------------------------------
leaderboard = []
for name, model in best_models.items():
    score = eval_model(model,
                       Xva_mat if USE_TFIDF_MATRIX else X_val,
                       y_val, TASK)
    leaderboard.append((name, score))
if ENSEMBLE_ON and ensemble_model is not None:
    score = eval_model(ensemble_model,
                       Xva_mat if USE_TFIDF_MATRIX else X_val,
                       y_val, TASK)
    leaderboard.append(("ensemble", score))

leaderboard.sort(key=lambda x: x[1], reverse = (TASK!="reg"))  # MAE lower-is-better
print(f"\n🏁 Validation leaderboard ({metric_name}):")
for rank,(n,s) in enumerate(leaderboard,1):
    print(f"{rank:>2}. {n:<10s} {s:.4f}")

# ── 4. Select BEST_KEY ----------------------------------------------------
BEST_KEY = leaderboard[0][0]      # auto-top; override if you like
BEST_MODEL = ensemble_model if BEST_KEY=="ensemble" else best_models[BEST_KEY]
print(f"\n✓ Selected winner → {BEST_KEY}")

# ── 5. Re-fit on FULL data & SAVE ----------------------------------------
if USE_TFIDF_MATRIX:
    Xfull = build_design_matrix(preproc, X_df, fit=True, tfidf_block=TFIDF_TRAIN)
    BEST_MODEL.fit(Xfull, y_vec)
else:
    BEST_MODEL.fit(X_df, y_vec)

joblib.dump(BEST_MODEL, "model.pkl")
print("💾 model.pkl saved.")

# ── 6. Inference on test & submission.csv ---------------------------------
if test_clean is not None:
    if USE_TFIDF_MATRIX:
        Xtest = build_design_matrix(preproc, test_clean, fit=False, tfidf_block=TFIDF_TEST)
    else:
        Xtest = test_clean
    preds = BEST_MODEL.predict(Xtest)
    # Formatting for multilabel: DataFrame with one column per label name
    if TASK=="multilabel":
        sub = pd.DataFrame(preds, columns=mlb.classes_)
    else:
        sub = pd.DataFrame({TARGET: preds})
    sub.to_csv("submission.csv", index=False)
    print("📝 submission.csv written.")
# ╚════════════════════════════════════════════════════════════════════════╝

</code></pre>

</details>

<details>
<summary>Best use cases (click to expand)</summary>

<pre><code class="language-python">
# ARCHITECTURES #

# Standalone: build_backbone(name, num_classes, in_ch, multilabel=False, pool='avg'|'gem')
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.tensor(p)); self.eps=eps
    def forward(self,x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),1).pow(1.0/self.p)

def _global_pool(kind): return nn.AdaptiveAvgPool2d(1) if kind=='avg' else GeM()

def _adapt_first_conv(conv: nn.Conv2d, in_ch: int, mode="avg"):
    w = conv.weight.data; out_ch, old_in, kH, kW = w.shape
    new = nn.Conv2d(in_ch, out_ch, (kH,kW), stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=1, bias=(conv.bias is not None),
                    padding_mode=conv.padding_mode)
    if in_ch == old_in: new.weight.data = w.clone()
    else:
        if mode=="avg": base = w.mean(1, keepdim=True).repeat(1,in_ch,1,1)
        else:           base = w.repeat(1, (in_ch+old_in-1)//old_in,1,1)[:, :in_ch]
        new.weight.data = base * (old_in / in_ch)
    if conv.bias is not None: new.bias.data = conv.bias.data.clone()
    return new

def _adapt_vit_in(model, in_ch:int):
    if hasattr(model, "conv_proj") and isinstance(model.conv_proj, nn.Conv2d):
        model.conv_proj = _adapt_first_conv(model.conv_proj, in_ch)
    else:
        for name,m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0]>1:
                obj=model; parts=name.split(".")
                for p in parts[:-1]: obj=getattr(obj,p)
                setattr(obj, parts[-1], _adapt_first_conv(m, in_ch))
                break
    return model

def build_backbone(name:str, num_classes:int, in_ch:int=3, pretrained=True,
                   multilabel=False, pool='avg', dropout=0.0):
    if not hasattr(tvm, name): raise ValueError(f"Unknown torchvision model: {name}")
    m = getattr(tvm, name)(weights="DEFAULT" if pretrained else None)

    # remove native classifiers & adapt first conv
    if name.startswith("resnet"):
        m.conv1 = _adapt_first_conv(m.conv1, in_ch)
        feat_dim = m.fc.in_features; m.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        m.features[0][0] = _adapt_first_conv(m.features[0][0], in_ch)
        feat_dim = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif name.startswith("convnext"):
        m.features[0][0] = _adapt_first_conv(m.features[0][0], in_ch)
        feat_dim = m.classifier[2].in_features; m.classifier = nn.Identity()
    elif name.startswith("vit"):
        m = _adapt_vit_in(m, in_ch); feat_dim = m.heads.head.in_features; m.heads = nn.Identity()
    else:
        raise ValueError("Add your model mapping here.")

    head = nn.Sequential(
        _global_pool('gem' if pool=='gem' else 'avg'), nn.Flatten(1),
        nn.Dropout(dropout) if dropout>0 else nn.Identity(),
        nn.Linear(feat_dim, num_classes)
    )

    class Net(nn.Module):
        def __init__(self, body, head): super().__init__(); self.body=body; self.head=head
        def forward(self,x):
            feats=self.body(x); 
            if feats.ndim==2: feats=feats[:, :, None, None]
            return self.head(feats)  # logits; apply sigmoid/softmax outside
    return Net(m, head)


# Standalone: build_unet(in_ch, out_ch, use_cbam=False, use_film=False, cond_dim=None)
import torch, torch.nn as nn, torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, ch, red=16, k=7):
        super().__init__()
        self.mlp  = nn.Sequential(nn.Linear(ch, ch//red), nn.ReLU(), nn.Linear(ch//red, ch))
        self.conv = nn.Conv2d(2, 1, k, padding=(k-1)//2)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        b,c,_,_ = x.shape
        ch_att = self.mlp(x.mean((2,3)).view(b,c)) + self.mlp(x.amax((2,3)).view(b,c))
        x = x * self.sig(ch_att).view(b,c,1,1)
        sp_att = self.sig(self.conv(torch.cat([x.mean(1,True), x.amax(1,True)],1)))
        return x * sp_att

class FiLM(nn.Module):
    def __init__(self, ch:int, cond_dim:int, hidden:int=128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, ch))
        self.b = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, ch))
    def forward(self, x, cond):
        if cond is None: return x
        gamma = self.g(cond).unsqueeze(-1).unsqueeze(-1)
        beta  = self.b(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, ci, co, use_attn=False):
        super().__init__()
        self.n1 = nn.GroupNorm(32, ci); self.c1 = nn.Conv2d(ci, co, 3,1,1)
        self.n2 = nn.GroupNorm(32, co); self.c2 = nn.Conv2d(co, co, 3,1,1)
        self.act = nn.SiLU(); self.skip = nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
        self.attn = CBAM(co) if use_attn else nn.Identity()
    def forward(self, x):
        h = self.act(self.n1(x)); h = self.act(self.n2(self.c1(h)))
        h = self.c2(h) + self.skip(x); return self.attn(h)

class Down(nn.Module):
    def __init__(self, ci, co, use_attn=False):
        super().__init__(); self.r1=ResBlock(ci,co,use_attn); self.r2=ResBlock(co,co,use_attn)
        self.down = nn.Conv2d(co, co, 3,2,1)
    def forward(self,x): h=self.r1(x); h=self.r2(h); return h, self.down(h)

class Up(nn.Module):
    def __init__(self, ci, co, use_attn=False):
        super().__init__(); self.up=nn.ConvTranspose2d(ci, ci, 4,2,1)
        self.conv=nn.Conv2d(ci+co, co, 3,1,1); self.res=ResBlock(co,co,use_attn)
    def forward(self,x,skip): x=self.up(x); x=torch.cat([x,skip],1); return self.res(self.conv(x))

class ResUNetCBAM(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, chs=(64,128,256), use_cbam=False, use_film=False, cond_dim=None):
        super().__init__(); self.use_film = use_film
        self.stem = nn.Conv2d(in_ch, chs[0], 3,1,1)
        self.enc1 = Down(chs[0], chs[0], use_cbam)
        self.enc2 = Down(chs[0], chs[1], use_cbam)
        self.mid  = ResBlock(chs[1], chs[2], use_cbam)
        self.up1  = Up(chs[2], chs[1], use_cbam)
        self.up2  = Up(chs[1], chs[0], use_cbam)
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        if use_film:
            assert cond_dim is not None, "set cond_dim when use_film=True"
            self.film1 = FiLM(chs[0], cond_dim); self.film2 = FiLM(chs[2], cond_dim)
    def forward(self, x, cond=None):
        s0 = self.stem(x); 
        if self.use_film: s0 = self.film1(s0, cond)
        h1, x1 = self.enc1(s0); h2, x2 = self.enc2(x1)
        m = self.mid(x2); 
        if self.use_film: m = self.film2(m, cond)
        u1 = self.up1(m, h2); u2 = self.up2(u1, h1)
        return self.head(u2)

def build_unet(in_ch:int, out_ch:int, use_cbam=False, use_film=False, cond_dim=None, chs=(64,128,256)):
    return ResUNetCBAM(in_ch=in_ch, out_ch=out_ch, chs=chs, use_cbam=use_cbam,
                       use_film=use_film, cond_dim=cond_dim)

unet = build_unet(in_ch=3, out_ch=1, use_cbam=False)            # default
unet_film = build_unet(in_ch=3, out_ch=1, use_cbam=False, use_film=True, cond_dim=8)

# Standalone: build_deeplab(num_classes, in_ch=3, backbone='resnet50'|'resnet101')
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

def _adapt_first_conv(conv: nn.Conv2d, in_ch: int):
    import torch
    w=conv.weight.data; out_ch, old_in, kH, kW = w.shape
    new = nn.Conv2d(in_ch, out_ch, (kH,kW), stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None))
    if in_ch==old_in: new.weight.data=w.clone()
    else: new.weight.data = w.mean(1, keepdim=True).repeat(1,in_ch,1,1)*(old_in/in_ch)
    if conv.bias is not None: new.bias.data = conv.bias.data.clone()
    return new

def build_deeplab(num_classes:int, in_ch:int=3, backbone="resnet50", pretrained=True):
    m = deeplabv3_resnet50(weights="DEFAULT") if backbone=="resnet50" else deeplabv3_resnet101(weights="DEFAULT")
    if in_ch != 3: m.backbone.conv1 = _adapt_first_conv(m.backbone.conv1, in_ch)
    m.classifier[-1] = nn.Conv2d(m.classifier[-1].in_channels, num_classes, 1)
    return m
# Standalone: build_detector(model='fasterrcnn_resnet50_fpn'|'retinanet_resnet50_fpn', num_classes, anchor_sizes=None)
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def build_detector(model="fasterrcnn_resnet50_fpn", num_classes=2, pretrained=True, anchor_sizes=(32,64,128,256,512)):
    if model.startswith("fasterrcnn"):
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, box_detections_per_img=300)
        in_feat = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)  # includes background
        if anchor_sizes is not None:
            m.anchor_generator.sizes = tuple([(s,) for s in anchor_sizes])
        return m
    elif model.startswith("retinanet"):
        m = retinanet_resnet50_fpn(weights="DEFAULT" if pretrained else None)
        n_anchors = m.head.classification_head.num_anchors
        in_ch = m.backbone.out_channels
        m.head.classification_head = RetinaNetClassificationHead(in_ch, n_anchors, num_classes-1)  # RetinaNet excludes background in head
        if anchor_sizes is not None:
            m.anchor_generator.sizes = tuple([(s,) for s in anchor_sizes])
        return m
    else:
        raise ValueError("Supported: 'fasterrcnn_resnet50_fpn' | 'retinanet_resnet50_fpn'")

#Augmentations

# B1 — build_classification_transforms
# Purpose: quick train/valid pipelines for image classification that just work.
# Use: t_train, t_valid = build_classification_transforms(224, strong=False)

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_classification_transforms(img_size=224, strong=False, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if strong:
        # Strong policy: TrivialAugmentWide (robust & fast), plus RRC
        train = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        # Light policy: stable for most tabular-ish CV tasks
        train = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    valid = T.Compose([
        T.Resize(int(img_size*1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, valid
# B2 — Cutout transform
# Purpose: erase random square patch(es); reduces overfit; cheap.
# Use: add Cutout(...) just before ToTensor() or after (tensor mode supported).

import random, torch
import torchvision.transforms.functional as F

class Cutout:
    def __init__(self, n_holes=1, length=32, fill=0):
        self.n_holes, self.length, self.fill = n_holes, length, fill
    def __call__(self, img):
        tensor_mode = isinstance(img, torch.Tensor)
        if not tensor_mode:
            img = F.to_tensor(img)
        c, h, w = img.shape
        for _ in range(self.n_holes):
            y = random.randint(0, h - 1); x = random.randint(0, w - 1)
            y1 = max(0, y - self.length // 2); y2 = min(h, y1 + self.length)
            x1 = max(0, x - self.length // 2); x2 = min(w, x1 + self.length)
            img[:, y1:y2, x1:x2] = self.fill if isinstance(self.fill, (int,float)) else torch.tensor(self.fill).view(-1,1,1)
        return img if tensor_mode else F.to_pil_image(img)
# B3 — MixUp / CutMix collate
# Purpose: boost generalization, especially for multilabel or long-tail.
# Use: loader = DataLoader(dataset, batch_size=..., shuffle=True, collate_fn=MixupCutmix(...))

import torch, random

class MixupCutmix:
    def __init__(self, num_classes, alpha=0.2, prob=0.5, mode="mixup", multilabel=False):
        """
        mode: 'mixup' | 'cutmix' | 'both' (randomly picks per batch)
        multilabel: if True, y are float multi-hot vectors; else we one-hot them.
        """
        self.K, self.alpha, self.prob, self.mode, self.multilabel = num_classes, alpha, prob, mode, multilabel

    def _one_hot(self, y):
        return torch.nn.functional.one_hot(y, num_classes=self.K).float()

    def __call__(self, batch):
        # batch: list of (image_tensor, label)
        imgs = torch.stack([b[0] for b in batch])
        ys   = [b[1] for b in batch]
        y = torch.stack(ys) if self.multilabel else self._one_hot(torch.tensor(ys))

        if random.random() > self.prob:     # no mix
            return imgs, y

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        idx = torch.randperm(imgs.size(0))
        imgs2, y2 = imgs[idx], y[idx]

        choose = self.mode
        if self.mode == "both": choose = "mixup" if random.random() < 0.5 else "cutmix"

        if choose == "mixup":
            mixed_x = lam * imgs + (1 - lam) * imgs2
            mixed_y = lam * y    + (1 - lam) * y2
            return mixed_x, mixed_y

        # CutMix
        B, C, H, W = imgs.shape
        cx, cy = random.randint(0, W-1), random.randint(0, H-1)
        rw, rh = int(W * (1 - lam)**0.5), int(H * (1 - lam)**0.5)
        x1, x2 = max(0, cx - rw//2), min(W, cx + rw//2)
        y1, y2 = max(0, cy - rh//2), min(H, cy + rh//2)
        imgs[:, :, y1:y2, x1:x2] = imgs2[:, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2-x1)*(y2-y1) / (W*H))
        mixed_y = lam_adj * y + (1 - lam_adj) * y2
        return imgs, mixed_y
# B4 — SpecAugment (tensor-only, expects shape [C,H,W]: H=freq, W=time)
# Purpose: regularize spectrograms without raw-audio processing.
# Use: add SpecAugment(...) into your spectrogram image pipeline (after ToTensor, before Normalize).

import torch, random

class SpecAugment:
    def __init__(self, time_mask_param=30, freq_mask_param=12, n_time_masks=2, n_freq_masks=2, inplace=True):
        self.T, self.F = time_mask_param, freq_mask_param
        self.nT, self.nF = n_time_masks, n_freq_masks
        self.inplace = inplace
    def __call__(self, x):
        # x: Tensor [C,H,W] or [B,C,H,W]
        single = x.dim()==3
        if single: x = x.unsqueeze(0)
        x = x if self.inplace else x.clone()
        B, C, H, W = x.shape
        for b in range(B):
            # Freq masks
            for _ in range(self.nF):
                f = random.randint(0, self.F)
                f0 = random.randint(0, max(0, H - f))
                x[b, :, f0:f0+f, :] = 0
            # Time masks
            for _ in range(self.nT):
                t = random.randint(0, self.T)
                t0 = random.randint(0, max(0, W - t))
                x[b, :, :, t0:t0+t] = 0
        return x.squeeze(0) if single else x

# B5 — segmentation_transforms
# Purpose: apply identical geometry to image & mask; color jitter on image only.
# Use: t = segmentation_transforms(train=True, img_size=512); img_t, mask_t = t(img, mask)

import random
import torchvision.transforms.functional as F
from PIL import Image

class SegTransform:
    def __init__(self, train=True, img_size=512, scale_range=(0.75, 1.25), hflip_p=0.5, color_jitter=(0.2,0.2,0.2,0.1)):
        self.train, self.img_size = train, img_size
        self.scale_range, self.hflip_p = scale_range, hflip_p
        self.color_jitter = color_jitter

    def _color_jitter(self, img):
        b, c, s, h = self.color_jitter
        img = F.adjust_brightness(img, 1 + random.uniform(-b, b))
        img = F.adjust_contrast(img, 1 + random.uniform(-c, c))
        img = F.adjust_saturation(img, 1 + random.uniform(-s, s))
        img = F.adjust_hue(img, random.uniform(-h, h))
        return img

    def __call__(self, img, mask):
        # img, mask: PIL Images or tensors; convert to PIL for functional ops
        pil_in = isinstance(img, Image.Image)
        if not pil_in:
            img = F.to_pil_image(img); mask = F.to_pil_image(mask)

        # random resize (scale jitter) then center crop
        if self.train:
            scale = random.uniform(*self.scale_range)
            new_sz = int(self.img_size * scale)
            img = F.resize(img, new_sz, interpolation=Image.BILINEAR)
            mask= F.resize(mask,new_sz, interpolation=Image.NEAREST)
            # random crop to target size
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.img_size, self.img_size))
            img = F.crop(img, i, j, h, w); mask = F.crop(mask, i, j, h, w)
            # random horizontal flip
            if random.random() < self.hflip_p:
                img = F.hflip(img); mask = F.hflip(mask)
            # light color jitter on image only
            img = self._color_jitter(img)
        else:
            # deterministic resize + center crop for validation
            img = F.resize(img, self.img_size, interpolation=Image.BILINEAR)
            mask= F.resize(mask,self.img_size, interpolation=Image.NEAREST)

        # to tensor
        img_t  = F.to_tensor(img)
        mask_t = torch.from_numpy(np.array(mask, dtype='int64')) if mask.mode != 'L' else torch.as_tensor(np.array(mask, dtype='int64'))
        # If mask is already {0..K-1}, keep as long tensor; otherwise threshold outside.
        return img_t, mask_t

                                 
#Loss Rank
# C1 — CE (with label smoothing) and Focal Loss for single-label classification
import torch, torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLS(nn.Module):
    """Cross-entropy with label smoothing. Use for single-label tasks."""
    def __init__(self, smoothing=0.05, weight=None, ignore_index=-100):
        super().__init__()
        self.smoothing = float(smoothing)
        self.weight = torch.tensor(weight) if isinstance(weight, (list, tuple)) else weight
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        # logits: (B, K), target: (B,) int64
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index)
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            mask = target != self.ignore_index
            true_dist[mask, target[mask]] = 1.0 - self.smoothing
        logp = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            w = self.weight.to(logits.device).unsqueeze(0)
            logp = logp * w
        loss = -(true_dist * logp).sum(dim=1)
        return loss[target != self.ignore_index].mean()

class FocalLoss(nn.Module):
    """Multi-class focal loss. Good for class imbalance or hard examples."""
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = None if alpha is None else torch.tensor(alpha)  # per-class weights
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: (B,K), target: (B,)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, target.view(-1,1)).clamp_(1e-8, 1.0)
        loss = -((1 - pt) ** self.gamma) * logp.gather(1, target.view(-1,1)).squeeze(1)
        if self.alpha is not None:
            a = self.alpha.to(logits.device)[target]
            loss = loss * a
        return loss.mean() if self.reduction == "mean" else loss.sum()
# CE with smoothing (safe default)
criterion = CrossEntropyLS(smoothing=0.05, weight=None)

# Focal for heavy imbalance (optionally pass per-class alpha list/ndarray)
criterion = FocalLoss(gamma=2.0, alpha=None)

# C2 — BCEWithLogits + Focal (binary) + ASL (Asymmetric Loss) for multi-label
import torch, torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLossPosWeight(nn.Module):
    """Standard BCE with optional positive class weights (pos_weight per class)."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)
    def forward(self, logits, targets):
        # logits, targets: (B, K) float
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=None if self.pos_weight is None else self.pos_weight.to(logits.device)
        )

class FocalBinaryLoss(nn.Module):
    """Binary focal loss applied per class for multi-label tasks."""
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
    def forward(self, logits, targets):
        # logits, targets: (B, K)
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean() if self.reduction=="mean" else loss.sum()

class AsymmetricLossMultiLabel(nn.Module):
    """
    ASL: better than focal on long-tailed multi-label.
    clip: clamp negatives to reduce easy-negatives domination.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gp, self.gn, self.clip, self.eps, self.red = gamma_pos, gamma_neg, clip, eps, reduction
    def forward(self, logits, targets):
        x = logits; y = targets
        x_sig = torch.sigmoid(x)
        if self.clip is not None and self.clip > 0:
            xn = (1 - x_sig).clamp(min=self.eps)
            x_sig = torch.where(y < 0.5, x_sig + self.clip, x_sig)
            x_sig = x_sig.clamp(0, 1)
        pt = x_sig * y + (1 - x_sig) * (1 - y)
        one_sided_gamma = self.gp * y + self.gn * (1 - y)
        loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        loss *= (1 - pt).pow(one_sided_gamma)
        return loss.mean() if self.red=="mean" else loss.sum()
# Plain multilabel
criterion = BCEWithLogitsLossPosWeight()

# Long-tail multilabel
criterion = AsymmetricLossMultiLabel(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
# or
criterion = FocalBinaryLoss(gamma=2.0, alpha=0.25)

# C3 — Segmentation losses; works for binary and multi-class (channel-first logits).
import torch, torch.nn as nn
import torch.nn.functional as F

def _to_one_hot(labels, num_classes, ignore_index=None):
    # labels: (B,H,W) int64; returns (B,C,H,W) one-hot (ignoring 'ignore_index')
    B, H, W = labels.shape
    oh = torch.zeros(B, num_classes, H, W, device=labels.device, dtype=torch.float32)
    if ignore_index is not None:
        mask = labels != ignore_index
        oh.scatter_(1, labels.clamp_min(0).unsqueeze(1), 1.0)
        oh *= mask.unsqueeze(1)
    else:
        oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh

class DiceLoss(nn.Module):
    """Binary or multi-class Dice. Expect logits: (B,C,H,W); targets: (B,H,W) or one-hot (B,C,H,W)."""
    def __init__(self, eps=1e-6, ignore_index=None, binary=False):
        super().__init__(); self.eps, self.ignore, self.binary = eps, ignore_index, binary
    def forward(self, logits, targets):
        if self.binary:
            probs = torch.sigmoid(logits)
            if targets.ndim == 3: targets = targets.float().unsqueeze(1)
        else:
            probs = torch.softmax(logits, dim=1)
            if targets.ndim == 3:
                C = logits.size(1); targets = _to_one_hot(targets.long(), C, self.ignore)
        dims = (0,2,3)
        num = 2 * (probs*targets).sum(dim=dims)
        den = (probs*probs).sum(dim=dims) + (targets*targets).sum(dim=dims) + self.eps
        dice = 1 - (num / den)
        return dice.mean()

class TverskyLoss(nn.Module):
    """Generalized Dice; alpha penalizes FN, beta penalizes FP."""
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-6, binary=False):
        super().__init__(); self.a, self.b, self.eps, self.binary = alpha, beta, eps, binary
    def forward(self, logits, targets):
        if self.binary:
            p = torch.sigmoid(logits); t = targets.float().unsqueeze(1) if targets.ndim==3 else targets
        else:
            p = torch.softmax(logits, dim=1)
            if targets.ndim==3:
                C = logits.size(1); t = _to_one_hot(targets.long(), C)
            else: t = targets
        dims=(0,2,3); TP=(p*t).sum(dims); FP=(p*(1-t)).sum(dims); FN=((1-p)*t).sum(dims)
        tversky = (TP + self.eps) / (TP + self.a*FN + self.b*FP + self.eps)
        return 1 - tversky.mean()

class FocalTverskyLoss(nn.Module):
    """Focal Tversky: add gamma to focus on hard pixels."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, binary=False):
        super().__init__(); self.base=TverskyLoss(alpha,beta,binary=binary); self.g=gamma
    def forward(self, logits, targets):
        t = 1 - self.base(logits, targets)   # tversky index
        return (1 - t)**self.g

class ComboSegLoss(nn.Module):
    """
    Combine CE/BCE with Dice/Tversky.
    Examples:
      ComboSegLoss("ce+dice", ce_weight=0.5)      # multi-class
      ComboSegLoss("bce+dice", bce_weight=0.7)    # binary
    """
    def __init__(self, mode="ce+dice", ce_weight=0.5, bce_weight=0.7, dice_kwargs=None, tversky_kwargs=None):
        super().__init__()
        self.mode = mode
        self.ce_w, self.bce_w = ce_weight, bce_weight
        self.dice = DiceLoss(**(dice_kwargs or {}))
        self.tversky = TverskyLoss(**(tversky_kwargs or {}))
    def forward(self, logits, targets):
        if self.mode == "ce+dice":
            ce = F.cross_entropy(logits, targets.long())
            return self.ce_w*ce + (1-self.ce_w)*self.dice(logits, targets)
        if self.mode == "bce+dice":
            bce = F.binary_cross_entropy_with_logits(logits, targets.float().unsqueeze(1) if targets.ndim==3 else targets.float())
            return self.bce_w*bce + (1-self.bce_w)*self.dice(logits, targets)
        if self.mode == "ce+tversky":
            ce = F.cross_entropy(logits, targets.long())
            return self.ce_w*ce + (1-self.ce_w)*self.tversky(logits, targets)
        raise ValueError("mode must be 'ce+dice' | 'bce+dice' | 'ce+tversky'")
# Binary seg: BCE + Dice
criterion = ComboSegLoss("bce+dice", bce_weight=0.7, dice_kwargs=dict(binary=True))

# Multi-class seg: CE + Dice (ignore index optional)
criterion = ComboSegLoss("ce+dice", ce_weight=0.5, dice_kwargs=dict(binary=False))

# C4 — LossComposer: sum arbitrary losses with weights; returns total and components
import torch.nn as nn

class LossComposer(nn.Module):
    """
    Example:
      loss = LossComposer([
          ("ce",  0.5, CrossEntropyLS(smoothing=0.05)),
          ("dice",0.5, DiceLoss(binary=False))
      ])
      total, parts = loss(logits, targets)
    """
    def __init__(self, items):
        super().__init__()
        self.items = nn.ModuleList([m for _,_,m in items])
        self.names = [n for n,_,_ in items]
        self.wgts  = [w for _,w,_ in items]
    def forward(self, *args, **kwargs):
        total = 0.0; parts = {}
        for name, w, mod in zip(self.names, self.wgts, self.items):
            val = mod(*args, **kwargs)
            parts[name] = float(val.detach().cpu())
            total = total + w * val
        return total, parts
# Multi-label: BCE + ASL blend
loss = LossComposer([
    ("bce", 0.7, BCEWithLogitsLossPosWeight()),
    ("asl", 0.3, AsymmetricLossMultiLabel())
])
total, parts = loss(logits, targets)

# C5 — Class weight utilities
import numpy as np
import torch

def class_weights_balanced(y_counts):
    """
    Balanced weights for CE: w_c = N / (K * n_c).
    y_counts: array-like length K with per-class counts.
    """
    y_counts = np.asarray(y_counts, dtype=np.float64)
    N, K = y_counts.sum(), len(y_counts)
    w = N / (K * np.maximum(y_counts, 1))
    return torch.tensor(w, dtype=torch.float32)

def class_weights_effective_number(y_counts, beta=0.9999):
    """
    From 'Class-Balanced Loss Based on Effective Number of Samples' (CVPR'19).
    Use as 'weight' in CE (C1) or 'alpha' (per class) in Focal (C1).
    """
    y_counts = np.asarray(y_counts, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, y_counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    w = w / w.sum() * len(y_counts)  # normalize around 1
    return torch.tensor(w, dtype=torch.float32)

def pos_weight_for_bce(y_binary):
    """
    pos_weight per class for BCEWithLogits:
      pos_weight[c] = (N_neg / N_pos).
    y_binary: ndarray/tensor shape (N, K) in {0,1}.
    """
    y = torch.as_tensor(y_binary, dtype=torch.float32)
    pos = y.sum(dim=0).clamp(min=1.0)
    neg = (y.shape[0] - pos).clamp(min=1.0)
    return (neg / pos).float()
# Single-label CE weights from counts
w = class_weights_effective_number(y_counts)      # or class_weights_balanced
criterion = CrossEntropyLS(smoothing=0.05, weight=w)

# Multi-label BCE pos_weight from label matrix
pos_w = pos_weight_for_bce(Y_train_bin)           # (K,)
criterion = BCEWithLogitsLossPosWeight(pos_weight=pos_w)

#Optimizers and schedulers 
# D1 — build_optimizer: AdamW / SGD with proper no-decay groups
import torch, torch.nn as nn

def _param_groups(model, weight_decay=1e-4, no_decay_modules=(nn.BatchNorm1d, nn.BatchNorm2d,
                                                             nn.BatchNorm3d, nn.GroupNorm,
                                                             nn.LayerNorm, nn.InstanceNorm1d,
                                                             nn.InstanceNorm2d, nn.InstanceNorm3d)):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = name.endswith(".bias")
        # any parameter inside normalization layers should skip weight decay
        in_norm = any(isinstance(m, no_decay_modules) and any(id(p) is id(pp) for pp in m.parameters(recurse=False))
                      for m in [])
        # fallback: detect by name (common in torchvision)
        if is_bias or ("norm" in name) or ("bn" in name) or ("ln" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_optimizer(model, name="adamw", lr=3e-4, weight_decay=1e-4, momentum=0.9, nesterov=True):
    """
    name: 'adamw' | 'sgd'
    Returns: torch.optim.Optimizer
    """
    groups = _param_groups(model, weight_decay)
    if name.lower() == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    if name.lower() == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=nesterov)
    raise ValueError("name must be 'adamw' or 'sgd'")
# clf/seg/det all fine
# opt = build_optimizer(model, name="adamw", lr=3e-4, weight_decay=1e-4)

# D2 — WarmupCosineLR: linear warmup → cosine anneal (per-iteration stepping)
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    """
    total_steps: number of optimizer.step() across training
    warmup_steps: linear warmup steps (<= total_steps)
    min_lr_ratio: final_lr = base_lr * min_lr_ratio
    """
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.0, last_epoch=-1):
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # 0-based inside scheduler
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * step / self.warmup_steps
            else:
                t = min(max(step - self.warmup_steps, 0), max(self.total_steps - self.warmup_steps, 1))
                T = max(self.total_steps - self.warmup_steps, 1)
                cos = 0.5 * (1 + math.cos(math.pi * t / T))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cos)
            lrs.append(lr)
        return lrs
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
sched = WarmupCosineLR(opt, total_steps=total_steps, warmup_steps=int(0.05*total_steps), min_lr_ratio=0.01)

# training loop: call per-iteration AFTER optimizer.step()
# scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

# D3 — OneCycleLR builder (per-iteration)
from torch.optim.lr_scheduler import OneCycleLR

def build_onecycle(opt, max_lr, steps_per_epoch, epochs, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    return OneCycleLR(opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
                      pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor)

sched = build_onecycle(opt, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
# call sched.step() every iteration after optimizer.step()

# D4 — ReduceLROnPlateau (per-epoch)
def build_plateau(opt, mode="max", factor=0.5, patience=2, min_lr=1e-6, threshold=1e-3):
    """
    mode: 'max' for metrics like F1/mAP; 'min' for losses/MAE.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr, verbose=True
    )

# D5 — clip gradients safely (use right before optimizer.step())
import torch

def clip_grad_norm_(model, max_norm=1.0, norm_type=2):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=norm_type)
# loss.backward()
# clip_grad_norm_(model, max_norm=1.0)
# optimizer.step()

# D6 — EMA of model params (update every iteration; swap weights for eval)
import copy, torch

class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device) if device else p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad: 
                continue
            assert name in self.shadow
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Copy EMA weights into the live model (e.g., before validation)."""
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])

    def state_dict(self):  # optional persistence
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}
    def load_state_dict(self, state):
        self.decay = state["decay"]; self.shadow = {k: v for k, v in state["shadow"].items()}

# ema = EMA(model, decay=0.999)

# # train loop (per iteration)
# # loss.backward(); clip_grad_norm_(...); optimizer.step(); optimizer.zero_grad()
# ema.update(model)

# # validation
# backup = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
# ema.apply_to(model)
# # ... run eval ...
# # restore
# for (n, p) in model.named_parameters():
#     if p.requires_grad: p.data.copy_(backup[n])

#TRAINING LOOP

# E0 — meters, early-stop, checkpoints
import time, math, os, copy, torch

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.n=0; self.sum=0.0
    def update(self, val, k=1): self.sum += float(val)*k; self.n += k
    @property
    def avg(self): return self.sum / max(self.n, 1)

class EarlyStopper:
    def __init__(self, patience=5, mode="max", min_delta=1e-6):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = -float("inf") if mode=="max" else float("inf")
        self.count = 0
    def step(self, value):
        improved = (value > self.best + self.min_delta) if self.mode=="max" else (value < self.best - self.min_delta)
        if improved: self.best=value; self.count=0; return False
        self.count += 1; return self.count > self.patience

def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model.state_dict()}
    if optimizer: payload["optimizer"] = optimizer.state_dict()
    if epoch is not None: payload["epoch"] = epoch
    if extra is not None: payload["extra"] = extra
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    return state.get("epoch", None), state.get("extra", None)
# E1 — forward+loss per task
import torch.nn.functional as F

def step_classification(model, batch, criterion, device, multilabel=False):
    """
    batch: (images, targets) where:
      - single-label: targets LongTensor [B]
      - multilabel : targets FloatTensor [B,K] (multi-hot)
    returns: loss, logits
    """
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
    loss = criterion(logits, y if multilabel else y.long())
    return loss, logits

def step_segmentation(model, batch, criterion, device):
    """
    batch: (images, masks) with masks LongTensor [B,H,W] for multiclass
           or Float/Long ([B,1,H,W] or [B,H,W]) for binary (criterion handles it).
    """
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
    loss = criterion(logits, y)
    return loss, logits

def step_detection(model, batch, device):
    """
    batch: (images_list, targets_list) as required by torchvision detection models.
    returns: loss, None
    """
    images, targets = batch
    images  = [im.to(device, non_blocking=True) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)  # model computes its own losses
    loss = sum(v for v in loss_dict.values())
    return loss, None
# E2 — metrics (no sklearn)
import torch

def f1_macro_from_logits(logits, y_true):
    """
    Single-label multi-class F1-macro from logits and int64 targets.
    """
    y_pred = logits.argmax(dim=1)
    K = int(max(int(y_true.max()), int(y_pred.max())) + 1)
    cm = torch.zeros((K, K), dtype=torch.int64, device=logits.device)
    cm.index_put_((y_true, y_pred), torch.ones_like(y_true, dtype=torch.int64), accumulate=True)

    # per-class precision/recall
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    f1 = (2*tp) / (2*tp + fp + fn + 1e-12)
    return torch.nanmean(f1).item()

def f1_micro_sigmoid(logits, y_true, thresh=0.5):
    """
    Multi-label F1-micro using thresholded sigmoid.
    logits: [B,K]; y_true: [B,K] in {0,1}
    """
    p = (torch.sigmoid(logits) >= thresh).float()
    tp = (p * y_true).sum()
    fp = (p * (1 - y_true)).sum()
    fn = ((1 - p) * y_true).sum()
    return (2*tp / (2*tp + fp + fn + 1e-12)).item()

def dice_iou_from_logits(logits, y_true, num_classes=None, is_binary=False, thresh=0.5):
    """
    For segmentation:
      - binary: logits [B,1,H,W], y_true [B,H,W] in {0,1} → dice, iou
      - multi : logits [B,C,H,W], y_true [B,H,W] in {0..C-1} → mean dice, mean iou (macro)
    """
    if is_binary:
        probs = torch.sigmoid(logits)
        preds = (probs >= thresh).float()
        y = y_true.float().unsqueeze(1)
        inter = (preds * y).sum((0,2,3))
        union = preds.sum((0,2,3)) + y.sum((0,2,3))
        dice = (2*inter + 1e-6) / (union + 1e-6)
        iou  = (inter + 1e-6) / (union - inter + 1e-6)
        return dice.mean().item(), iou.mean().item()
    else:
        C = logits.size(1) if num_classes is None else num_classes
        preds = logits.argmax(1)
        dice_list, iou_list = [], []
        for c in range(C):
            p = (preds == c).float()
            t = (y_true == c).float()
            inter = (p*t).sum()
            union = p.sum() + t.sum()
            dice_list.append(((2*inter + 1e-6) / (union + 1e-6)).item())
            iou_list.append(((inter + 1e-6) / (p.sum() + t.sum() - inter + 1e-6)).item())
        return sum(dice_list)/C, sum(iou_list)/C

# Simple mAP@0.5 (class-agnostic micro AP, or per-class averaged if labels present)
def map50(preds, gts, iou_thresh=0.5):
    """
    preds: list of dicts per image: {'boxes': Tensor[N,4], 'scores': Tensor[N], 'labels': Tensor[N]}
    gts  : list of dicts per image: {'boxes': Tensor[M,4], 'labels': Tensor[M]}
    Returns micro AP@0.5 over all classes (or macro over classes if labels are present).
    """
    # Flatten all predictions with (img_id, score, label, box)
    flat = []
    for i, p in enumerate(preds):
        boxes = p["boxes"]; scores = p["scores"]; labels = p.get("labels", torch.zeros(len(boxes), dtype=torch.long))
        for b, s, l in zip(boxes, scores, labels):
            flat.append((i, float(s), int(l), b))
    flat.sort(key=lambda x: -x[1])  # by score desc

    # Build gt flags (per image, per class)
    gt_used = [torch.zeros(len(gt["boxes"]), dtype=torch.bool) for gt in gts]

    def iou(a, b):
        # boxes: [4] xyxy
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
        sa = max(0, xa2-xa1) * max(0, ya2-ya1); sb = max(0, xb2-xb1) * max(0, yb2-yb1)
        return inter / (sa + sb - inter + 1e-12)

    tp, fp, npos = [], [], 0
    # count positives
    for gt in gts: npos += len(gt["boxes"])

    for img_id, score, label, box in flat:
        gt_boxes = gts[img_id]["boxes"]; gt_labels = gts[img_id].get("labels", torch.zeros(len(gt_boxes), dtype=torch.long))
        # match to best IoU GT of same class (if labels present), else any
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if gt_used[img_id][j]: continue
            if "labels" in gts[img_id] and int(gt_labels[j]) != int(label): continue
            iou_ = iou(box.tolist(), gb.tolist())
            if iou_ > best_iou: best_iou, best_j = iou_, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp.append(1.0); fp.append(0.0); gt_used[img_id][best_j] = True
        else:
            tp.append(0.0); fp.append(1.0)

    if len(tp) == 0:
        return 0.0
    import numpy as np
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    rec = tp / max(npos, 1e-12)
    prec = tp / np.maximum(tp + fp, 1e-12)
    # 11-point interpolation
    ap = 0.0
    for r in [i/10 for i in range(11)]:
        p = np.max(prec[rec >= r]) if np.any(rec >= r) else 0.0
        ap += p / 11.0
    return float(ap)
# E3 — fit_one_epoch / validate
from contextlib import nullcontext

def fit_one_epoch(model, loader, optimizer, criterion, device, task,
                  scaler=None, accum_steps=1, scheduler=None, sched_step="step",
                  ema=None, print_freq=50, multilabel=False, clip_grad_norm=None):
    model.train()
    loss_meter = AvgMeter()
    step_fn = step_detection if task=="det" else (lambda b: step_segmentation(model, b, criterion, device) if task=="seg"
                                                  else step_classification(model, b, criterion, device, multilabel))
    autocast = torch.cuda.amp.autocast if (scaler is not None) else nullcontext
    optimizer.zero_grad(set_to_none=True)
    for it, batch in enumerate(loader, 1):
        with autocast():
            loss, _ = step_fn(batch)
        loss_meter.update(loss.item(), 1)

        if scaler is not None:
            scaler.scale(loss).backward()
            if it % accum_steps == 0:
                if clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and sched_step=="step": scheduler.step()
        else:
            loss.backward()
            if it % accum_steps == 0:
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and sched_step=="step": scheduler.step()

        if ema is not None:
            ema.update(model)

        if (it % print_freq == 0) or (it == len(loader)):
            print(f"  it {it:5d}/{len(loader)} | loss {loss_meter.avg:.4f}")

    return loss_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device, task, ema=None,
             multilabel=False, thr=0.5, num_classes=None):
    model_was = None
    # Optionally evaluate EMA weights
    if ema is not None:
        model_was = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
    model.eval()

    loss_meter = AvgMeter()
    if task == "det":
        preds_all, gts_all = [], []
        for batch in loader:
            images, targets = batch
            images = [im.to(device) for im in images]
            outputs = model(images)   # list of dicts
            preds_all.extend([{k: v.detach().cpu() for k, v in o.items()} for o in outputs])
            gts_all.extend([{k: v.detach().cpu() for k, v in t.items()} for t in targets])
        score = map50(preds_all, gts_all)
        metric_name, metric_value = "mAP@0.5", score
        val_loss = math.nan
    else:
        # clf/seg need criterion for loss
        preds_list, targets_list = [], []
        for batch in loader:
            if task == "seg":
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss = criterion(logits, y).item()
                loss_meter.update(val_loss, x.size(0))
                preds_list.append(logits.detach().cpu())
                targets_list.append(y.detach().cpu())
            else:  # classification
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss = criterion(logits, y if multilabel else y.long()).item()
                loss_meter.update(val_loss, x.size(0))
                preds_list.append(logits.detach().cpu())
                targets_list.append(y.detach().cpu())
        logits = torch.cat(preds_list, 0)
        y_true = torch.cat(targets_list, 0)
        if task == "seg":
            is_binary = (logits.size(1) == 1)
            dice, iou = dice_iou_from_logits(logits, y_true, num_classes=num_classes, is_binary=is_binary)
            metric_name, metric_value = ("mIoU" if not is_binary else "Dice"), (iou if not is_binary else dice)
        else:
            if multilabel:
                metric_name, metric_value = "F1-micro", f1_micro_sigmoid(logits, y_true.float(), thresh=thr)
            else:
                metric_name, metric_value = "F1-macro", f1_macro_from_logits(logits, y_true.long())

    # restore non-EMA weights if needed
    if ema is not None and model_was is not None:
        model.load_state_dict(model_was)

    return loss_meter.avg, metric_name, metric_value
# E4 — train loop (task-agnostic)
def train_loop(model, train_loader, valid_loader, optimizer, criterion, device,
               task, epochs=20, accum_steps=1, scheduler=None, sched_step="step",
               use_amp=True, ema=None, early_stop_patience=5, monitor_mode="max",
               monitor_metric=None, save_best_path="best.pt", save_last_path="last.pt",
               multilabel=False, thr=0.5, num_classes=None, clip_grad_norm=None):
    """
    monitor_metric: name to maximize/minimize; if None, defaults per task:
        - clf single: 'F1-macro'
        - clf multi  : 'F1-micro'
        - seg binary : 'Dice'
        - seg multi  : 'mIoU'
        - det        : 'mAP@0.5'
    sched_step: 'step' (call after each optimizer.step) or 'epoch' (call at epoch end with metric/loss)
    """
    device = torch.device(device)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    stopper = EarlyStopper(patience=early_stop_patience, mode=monitor_mode)

    best_score = -float("inf") if monitor_mode=="max" else float("inf")
    best_epoch = -1

    # infer default metric name if not provided
    if monitor_metric is None:
        monitor_metric = "mAP@0.5" if task=="det" else ("Dice" if task=="seg" and (getattr(model, 'out_ch', None)==1 or True) else "F1-macro")
        if task=="seg" and (num_classes and num_classes>1): monitor_metric = "mIoU"
        if task=="clf" and multilabel: monitor_metric = "F1-micro"

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = fit_one_epoch(
            model, train_loader, optimizer, criterion, device, task,
            scaler=scaler, accum_steps=accum_steps, scheduler=scheduler, sched_step=sched_step,
            ema=ema, print_freq=50, multilabel=multilabel, clip_grad_norm=clip_grad_norm
        )

        val_loss, val_metric_name, val_metric_value = validate(
            model, valid_loader, criterion if task!="det" else None, device, task,
            ema=ema, multilabel=multilabel, thr=thr, num_classes=num_classes
        )

        # epoch-scheduler step if required
        if scheduler is not None and sched_step == "epoch":
            # Plateau expects a value (maximize metric or minimize loss)
            if hasattr(scheduler, "step"):
                if isinstance(val_metric_value, float) and (monitor_mode=="max"):
                    scheduler.step(val_metric_value)
                elif (monitor_mode=="min"):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # monitor & early stop
        monitored = val_metric_value if monitor_mode=="max" else val_loss if monitor_metric.lower().endswith("loss") else val_metric_value
        is_better = (monitored > best_score) if monitor_mode=="max" else (monitored < best_score)

        print(f"  train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {val_metric_name} {val_metric_value:.4f}")
        if is_better:
            best_score = monitored; best_epoch = epoch
            save_checkpoint(save_best_path, model, optimizer=None, epoch=epoch, extra={"metric": val_metric_value})
            print(f"  ✓ Saved best to {save_best_path}")

        should_stop = stopper.step(monitored)
        save_checkpoint(save_last_path, model, optimizer=None, epoch=epoch, extra={"metric": val_metric_value})

        if should_stop:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch}).")
            break

    print(f"\nBest {monitor_metric}: {best_score:.4f} at epoch {best_epoch}")
    return {"best_epoch": best_epoch, "best_score": best_score, "best_path": save_best_path}

# Example
## Classification (single-label)
# model = ...                      # any nn.Module returning logits [B,K]
# criterion = CrossEntropyLS(0.05) # from Section C1; or use F.cross_entropy directly
# opt = build_optimizer(model, "adamw", lr=3e-4)  # Section D1
# sched, step_mode = build_scheduler(opt, "warmup_cosine",
#                                    total_steps=len(train_loader)*EPOCHS,
#                                    warmup_steps=int(0.05*len(train_loader)*EPOCHS),
#                                    min_lr_ratio=0.01)
# ema = EMA(model, decay=0.999)    # Section D6 (optional)

# train_loop(model, train_loader, valid_loader, opt, criterion, "cuda",
#            task="clf", epochs=30, accum_steps=1, scheduler=sched, sched_step=step_mode,
#            ema=ema, early_stop_patience=5, monitor_mode="max",
#            multilabel=False, save_best_path="best_clf.pt")

# # Segmentation (binary)
# model = build_unet(in_ch=3, out_ch=1, use_cbam=False)  # from Section A2
# criterion = ComboSegLoss("bce+dice", bce_weight=0.7, dice_kwargs=dict(binary=True))  # Section C3
# opt = build_optimizer(model, "adamw", lr=3e-4)
# sched, step_mode = build_scheduler(opt, "warmup_cosine",
#                                    total_steps=len(train_loader)*EPOCHS,
#                                    warmup_steps=int(0.05*len(train_loader)*EPOCHS))
# train_loop(model, train_loader, valid_loader, opt, criterion, "cuda",
#            task="seg", epochs=60, scheduler=sched, sched_step=step_mode,
#            save_best_path="best_seg.pt", num_classes=None)

# # Detection (torchvision FasterRCNN)
# model = build_detector("fasterrcnn_resnet50_fpn", num_classes=5)  # Section A3
# opt = build_optimizer(model, "sgd", lr=0.01, weight_decay=1e-4)
# sched, step_mode = build_scheduler(opt, "plateau", mode="max", factor=0.5, patience=2)
# train_loop(model, train_loader, valid_loader, opt, criterion=None, device="cuda",
#            task="det", epochs=12, scheduler=sched, sched_step="epoch",
#            save_best_path="best_det.pt")

# CLASS COUNTS
# F1 — compute_class_counts
# Purpose: get per-class counts for weighting/sampling (fast & standalone).
import numpy as np
import torch

def class_counts_single(y):
    """
    y: 1D array/tensor of int labels [N]
    returns: counts (K,)
    """
    y = np.asarray(y)
    K = int(y.max()) + 1
    cnt = np.bincount(y, minlength=K).astype(np.int64)
    return cnt

def class_counts_multilabel(Y):
    """
    Y: array/tensor [N, K] in {0,1}
    returns: positive counts per class (K,)
    """
    Y = torch.as_tensor(Y).float()
    return Y.sum(dim=0).cpu().numpy().astype(np.int64)
## single-label
# cnt = class_counts_single(y_train)      # (K,)
## multilabel
# cnt = class_counts_multilabel(Y_train)  # (K,)

# F2 — class weighting utilities
import numpy as np
import torch

def weights_balanced(counts):
    """
    w_c = N / (K * n_c)  (safe default for CE)
    counts: (K,)
    """
    counts = np.asarray(counts, dtype=np.float64)
    N, K = counts.sum(), len(counts)
    w = N / (K * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)

def weights_effective_number(counts, beta=0.9999):
    """
    Class-Balanced Loss (CVPR'19): w_c ∝ (1 - beta) / (1 - beta^{n_c})
    Good when the tail is very long.
    """
    counts = np.asarray(counts, dtype=np.float64)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-8)
    w = w / w.sum() * len(counts)  # normalize around 1
    return torch.tensor(w, dtype=torch.float32)

def bce_pos_weight(Y):
    """
    pos_weight for BCEWithLogits: pos_weight[c] = N_neg / N_pos  (per class)
    Y: [N, K] in {0,1}
    """
    Y = torch.as_tensor(Y).float()
    pos = Y.sum(dim=0).clamp(min=1.0)
    neg = (Y.shape[0] - pos).clamp(min=1.0)
    return (neg / pos).float()
## CE (single-label)
# w = weights_effective_number(class_counts_single(y_train))   # or weights_balanced
## BCE (multilabel)
# pos_w = bce_pos_weight(Y_train)  # pass to F.binary_cross_entropy_with_logits(..., pos_weight=pos_w)

</code></pre>
</details>

<details>
<summary>🧠 Additional examples (click to expand)</summary>
<pre><code class="language-python">
  
!pip -q install --upgrade datasets transformers[torch] accelerate evaluate
!pip -q install sentence-transformers peft bitsandbytes  # ok to re-run

import torch, numpy as np, random, os, evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM,
    TrainingArguments, Trainer,
    DataCollatorWithPadding, DataCollatorForTokenClassification
)

# deterministic runs
def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- 2-liner save / load for checkpoints
def save_ckpt(model, tok, path:str):
    model.save_pretrained(path); tok.save_pretrained(path)

def load_ckpt(path:str, num_labels:int=2):
    tok  = AutoTokenizer.from_pretrained(path, use_fast=True)
    mdl  = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    return tok, mdl

device = "cuda" if torch.cuda.is_available() else "cpu"




# 1 ·  SEQUENCE / SENTENCE CLASSIFICATION
#     (single *or* pair-sentence tasks)

# 🔶 BASIC SETTINGS ------------------------------------------------
BASE_MODEL   = "bert-base-uncased"    # HF id of encoder
RESUME_FROM  = None                   # path to a saved ckpt to warm-start
NUM_LABELS   = 2                      # >2 for multi-class
# Dataset options:
#   – HF hub   : DATASET_NAME="glue", DATASET_CFG="sst2"
#   – local dir: DATASET_NAME="./my_csv", DATASET_CFG=None  (expects train.csv / dev.csv)
DATASET_NAME, DATASET_CFG = "glue", "sst2"
TEXT_KEY   = "sentence"               # column for single-text tasks
TEXT_KEY_A = None                     # for pair tasks  e.g. "question1"
TEXT_KEY_B = None                     # second sentence col
LABEL_KEY  = "label"                  # ground-truth class label
# 🔶 END SETTINGS --------------------------------------------------

# 1A ·  Load model / tokenizer
if RESUME_FROM:
    tokenizer, model = load_ckpt(RESUME_FROM, NUM_LABELS)
    print(f"✓ Resumed encoder - {RESUME_FROM}")
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL, num_labels=NUM_LABELS)

# 1B ·  Load dataset (hub or local)
if os.path.isdir(DATASET_NAME):
    cls_ds = load_dataset("csv",
        data_files={"train":f"{DATASET_NAME}/train.csv",
                    "validation":f"{DATASET_NAME}/dev.csv"})
else:
    cls_ds = load_dataset(DATASET_NAME, DATASET_CFG)
print("📑 available columns:", cls_ds["train"].column_names)

# 1C ·  Tokenise
def cls_tok(ex):
    if TEXT_KEY_A and TEXT_KEY_B:
        return tokenizer(ex[TEXT_KEY_A], ex[TEXT_KEY_B], truncation=True)
    return tokenizer(ex[TEXT_KEY], truncation=True)
cls_tok_ds = cls_ds.map(cls_tok, batched=True)

# 1D ·  Collator + metric
collator = DataCollatorWithPadding(tokenizer)        # pads to longest in mini-batch
metric   = evaluate.load("accuracy")
def compute_metrics(ep):
    preds = np.argmax(ep.predictions, axis=1)
    return metric.compute(predictions=preds, references=ep.label_ids)

# 1E ·  Train
args = TrainingArguments(
    output_dir="seqcls-run",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    weight_decay=0.01,
)
Trainer(model, args,
        train_dataset=cls_tok_ds["train"],
        eval_dataset=cls_tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics).train()
save_ckpt(model, tokenizer, "seqcls_finetuned")
print("✓ saved → ./seqcls_finetuned")




# 2 ·  TOKEN CLASSIFICATION  (NER / POS)  — WordPiece-aware

# 🔶 SETTINGS -----------------------------------------------------
NER_DS_NAME, NER_CFG = "conll2003", None          # HF id or local dir
TOKENS_KEY, TAGS_KEY = "tokens", "ner_tags"       # list[str], list[int]
RESUME_FOR_NER       = "seqcls_finetuned"         # use encoder from §1
# 🔶 END SETTINGS -------------------------------------------------

ner_ds = load_dataset(NER_DS_NAME, NER_CFG) if not os.path.isdir(NER_DS_NAME) else \
         load_dataset("json", data_files={s:f"{NER_DS_NAME}/{s}.json"
                                          for s in ["train","validation"]})

label_list = ner_ds["train"].features[TAGS_KEY].feature.names
id2label   = dict(enumerate(label_list)); label2id = {v:k for k,v in id2label.items()}

tokenizer  = AutoTokenizer.from_pretrained(RESUME_FOR_NER or BASE_MODEL, use_fast=True)
from transformers import BertForTokenClassification
ner_model  = BertForTokenClassification.from_pretrained(
                RESUME_FOR_NER or BASE_MODEL,
                id2label=id2label, label2id=label2id).to(device)

# --- label alignment helper
def align_labels(ex):
    tok = tokenizer(ex[TOKENS_KEY], is_split_into_words=True,
                    truncation=True, return_offsets_mapping=True)
    w_ids, labels, prev = tok.word_ids(), [], None
    for w in w_ids:
        if w is None: labels.append(-100)
        elif w != prev:
            labels.append(ex[TAGS_KEY][w])          # B-tag stays
        else:                                       # sub-token
            tag = label_list[ex[TAGS_KEY][w]]
            labels.append(ex[TAGS_KEY][w] if tag.startswith("I-") else -100)
        prev = w
    tok["labels"] = labels
    tok.pop("offset_mapping")
    return tok

ner_tok  = ner_ds.map(align_labels, batched=True, remove_columns=ner_ds["train"].column_names)
ner_coll = DataCollatorForTokenClassification(tokenizer)

seqeval  = evaluate.load("seqeval")
def ner_metrics(p):
    preds = np.argmax(p.predictions, -1)
    true_p, true_l = [], []
    for pr, lb in zip(preds, p.label_ids):
        tp, tl = [], []
        for p_i, l_i in zip(pr, lb):
            if l_i != -100:
                tp.append(label_list[p_i]); tl.append(label_list[l_i])
        true_p.append(tp); true_l.append(tl)
    return {"f1": seqeval.compute(predictions=true_p, references=true_l)["overall_f1"]}

ner_args = TrainingArguments("ner-run",
                             learning_rate=3e-5,
                             per_device_train_batch_size=8,
                             num_train_epochs=3,
                             evaluation_strategy="epoch",
                             fp16=torch.cuda.is_available())
Trainer(ner_model, ner_args,
        train_dataset=ner_tok["train"], eval_dataset=ner_tok["validation"],
        tokenizer=tokenizer, data_collator=ner_coll,
        compute_metrics=ner_metrics).train()
save_ckpt(ner_model, tokenizer, "ner_finetuned")
print("✓ saved → ./ner_finetuned")




# 3 ·  PRO MLM  — domain adaptation with flexible masking schemes

# 🔶 SETTINGS
PLAIN_TXT     = "./domain_corpus.txt"   # one sentence (or paragraph) per line
MODEL_CHECKPT = "bert-base-uncased"     # can be roberta-large, deberta, etc.
EPOCHS        = 2                       # 1 is fine for quick adapt
BSZ           = 32                      # per-device
LR            = 5e-5
MASK_STRATEGY = "wwm"                   # "standard" | "wwm" | "span"
MLM_PROB      = 0.15                    # % of (words or tokens) to corrupt
EVAL_STEPS    = 1000                    # perplexity logging
BLOCK_SIZE    = 128                     # set 0 to disable chunking
# 🔶 END

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorForPermutationLanguageModeling  # span / XLNet style
)
from itertools import islice
import math, evaluate

# 3A load corpus ─────────
raw_ds = load_dataset("text", data_files=PLAIN_TXT)["train"]

# 3B tokeniser + optional chunking ─────────
tok = AutoTokenizer.from_pretrained(MODEL_CHECKPT, use_fast=True)
def tok_line(ex):
    return tok(ex["text"], truncation=False)
if BLOCK_SIZE:
    # concatenate then split into BLOCK_SIZE chunks (as in RoBERTa pre-train scripts)
    def group(examples):
        concat = sum(examples["input_ids"], [])
        total  = (len(concat) // BLOCK_SIZE) * BLOCK_SIZE
        ids    = [concat[i : i + BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
        return {"input_ids": ids}
    token_ds = raw_ds.map(tok_line, batched=True, remove_columns=["text"])\
                     .map(group, batched=True)
else:
    token_ds = raw_ds.map(tok_line, batched=True, remove_columns=["text"])

# 3C choose masking collator ─────────
if MASK_STRATEGY == "wwm":
    collator = DataCollatorForWholeWordMask(tok, mlm_probability=MLM_PROB)
elif MASK_STRATEGY == "span":
    collator = DataCollatorForPermutationLanguageModeling(tok, plm_probability=MLM_PROB)
else:  # standard token-level
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=MLM_PROB)

model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPT).to(device)

# 3D compute perplexity callback ─────────
metric_ppl = evaluate.load("perplexity")
def eval_ppl(step):
    subset = token_ds.select(range(2048))        # small eval subset
    ppl = metric_ppl.compute(model=model, tokenizer=tok, dataset=subset)["perplexity"]
    print(f"🧮 step {step}: perplexity = {ppl:.2f}")

# 3E training loop ─────────
args = TrainingArguments(
    "mlm-run",
    learning_rate=LR,
    per_device_train_batch_size=BSZ,
    num_train_epochs=EPOCHS,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    logging_steps=EVAL_STEPS,
)

class PplCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % EVAL_STEPS == 0:
            eval_ppl(state.global_step)

trainer = Trainer(model, args,
                  train_dataset=token_ds,
                  data_collator=collator,
                  tokenizer=tok,
                  callbacks=[PplCallback])
trainer.train()

save_ckpt(model, tok, "mlm_adapt_full")
print("✓ full MLM adapt saved → ./mlm_adapt_full")



# 4 · PRO SimCSE  — unsup / supervised / hard-negatives in one place

# 🔶 SETTINGS
ENCODER_CKPT     = "mlm_adapt_full"     # reuse MLM-adapted weights
POOLING_STRATEGY = "mean"               # "cls" | "mean" | "max"
MODE             = "unsup"              # "unsup" | "sup" | "unsup_hard"
BATCH_SIZE       = 64
EPOCHS_SIMCSE    = 1
HARD_NEG_K       = 5                    # BM25 top-k if using hard neg mode
STS_VAL          = True                 # run STS-benchmark val every epoch?
# 🔶 END

from sentence_transformers import (SentenceTransformer, models,
                                   losses, SentenceDataset, evaluation)
from rank_bm25 import BM25Okapi

# 4A build base ST model ─────────
w_emb   = models.Transformer(ENCODER_CKPT, max_seq_length=128)
pooling = models.Pooling(w_emb.get_word_embedding_dimension(), POOLING_STRATEGY)
st_model = SentenceTransformer(modules=[w_emb, pooling])

# 4B create training pairs ─────────
raw_lines = [l.strip() for l in open(PLAIN_TXT, encoding="utf8")]
if MODE == "unsup":
    train_examples = [SentenceTransformer.InputExample(texts=[s, s]) for s in raw_lines]
elif MODE == "sup":                                     # expect label column tab-sep: sent1 \t sent2 \t score
    sup_pairs = [l.split("\t") for l in open("./simcse_sup.tsv")]
    train_examples = [SentenceTransformer.InputExample(texts=[p[0], p[1]], label=float(p[2]))
                      for p in sup_pairs]
else:  # unsup_hard
    bm25 = BM25Okapi([l.split() for l in raw_lines])
    train_examples = []
    for s in raw_lines:
        pos = s
        neg = " ".join(bm25.get_top_n(s.split(), raw_lines, n=HARD_NEG_K)[-1].split())
        train_examples.append(SentenceTransformer.InputExample(texts=[pos, neg]))

train_loader = torch.utils.data.DataLoader(train_examples,
                    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 4C loss selection ─────────
if MODE == "sup":
    loss_fn = losses.CosineSimilarityLoss(st_model)
else:
    loss_fn = losses.MultipleNegativesRankingLoss(st_model)

# 4D optional STS-benchmark evaluator ─────────
evaluator = None
if STS_VAL:
    sts = load_dataset("stsbenchmark", split="validation[:1000]")
    s1, s2, scr = sts["sentence1"], sts["sentence2"], sts["score"]
    evaluator = evaluation.EmbeddingSimilarityEvaluator(s1, s2, scr, batch_size=256)

# 4E train ─────────
st_model.fit(train_objectives=[(train_loader, loss_fn)],
             epochs=EPOCHS_SIMCSE,
             warmup_steps=int(0.1*len(train_loader)),
             evaluator=evaluator, evaluation_steps=500)

st_model.save("simcse_embedder_full")
print("✓ SimCSE saved → ./simcse_embedder_full")



# 5 ·  LoRA (4-bit) — wrap *any* head for low-VRAM fine-tune

from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
base_cls = AutoModelForSequenceClassification.from_pretrained(
              BASE_MODEL, num_labels=NUM_LABELS,
              quantization_config=bnb_cfg).to(device)

lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                      task_type="SEQ_CLS", target_modules=["query","value"])
lora_model = get_peft_model(base_cls, lora_cfg).to(device)
print(lora_model.print_trainable_parameters())   # should be ~<1 %

# use cls_tok_ds & collator from §1
lora_args = TrainingArguments("seqcls-lora",
                              learning_rate=3e-4,
                              num_train_epochs=3,
                              per_device_train_batch_size=16,
                              evaluation_strategy="steps",
                              eval_steps=200,
                              fp16=True)
Trainer(lora_model, lora_args,
        train_dataset=cls_tok_ds["train"],
        eval_dataset=cls_tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics).train()
lora_model.merge_and_unload()
save_ckpt(lora_model, tokenizer, "seqcls_lora_finetuned")
print("✓ LoRA ckpt → ./seqcls_lora_finetuned")





# 6 · MULTIPLE-CHOICE  (SWAG, HellaSwag, PIQA…)
#   *Model expects [bs, num_choices, seq_len]; collator does the reshaping.*

# 🔶 PARAMS
MC_DS_NAME, MC_CFG = "swag", None           # or "./mc_folder"
STEM1, STEM2   = "sent1", "sent2"           # base + follow-up context
ENDING_KEYS    = [f"ending{i}" for i in range(4)]
LABEL_KEY_MC   = "label"
RESUME_FOR_MC  = "seqcls_finetuned"         # encoder from Section 1 (optional)
# 🔶 END

from transformers import AutoModelForMultipleChoice, DataCollatorForMultipleChoice
mc_ds = load_dataset(MC_DS_NAME, MC_CFG) if not os.path.isdir(MC_DS_NAME) else \
        load_dataset("json", data_files={s:f"{MC_DS_NAME}/{s}.json" for s in ["train","validation"]})

tokenizer = AutoTokenizer.from_pretrained(RESUME_FOR_MC or BASE_MODEL, use_fast=True)
def mc_preprocess(ex):
    firsts  = [[s]*len(ENDING_KEYS) for s in ex[STEM1]]
    seconds = [[f"{ex[STEM2][i]} {ex[k][i]}" for k in ENDING_KEYS] for i in range(len(ex[STEM2]))]
    firsts, seconds = sum(firsts, []), sum(seconds, [])
    tok = tokenizer(firsts, seconds, truncation=True)
    # reshape back: list[len(keys)] → [bs, num_choices, seq_len]
    tok = {k:[v[i:i+len(ENDING_KEYS)] for i in range(0, len(v), len(ENDING_KEYS))]
           for k,v in tok.items()}
    tok["labels"] = ex[LABEL_KEY_MC]
    return tok

mc_tok = mc_ds.map(mc_preprocess, batched=True, remove_columns=mc_ds["train"].column_names)
mc_model = AutoModelForMultipleChoice.from_pretrained(RESUME_FOR_MC or BASE_MODEL).to(device)
mc_coll  = DataCollatorForMultipleChoice(tokenizer)

acc = evaluate.load("accuracy")
def mc_metrics(p):
    pred = np.argmax(p.predictions, -1)
    return acc.compute(predictions=pred, references=p.label_ids)

mc_args = TrainingArguments("mc-run", evaluation_strategy="epoch",
                            learning_rate=2e-5, per_device_train_batch_size=8,
                            num_train_epochs=3, fp16=torch.cuda.is_available())
Trainer(mc_model, mc_args,
        train_dataset=mc_tok["train"], eval_dataset=mc_tok["validation"],
        tokenizer=tokenizer, data_collator=mc_coll,
        compute_metrics=mc_metrics).train()
save_ckpt(mc_model, tokenizer, "mc_finetuned")
print("✓ Multiple-choice model saved at ./mc_finetuned")




# 7 · EXTRACTIVE QUESTION-ANSWERING  (SQuAD-style)
#   *Uses sliding window to handle long contexts.*

# 🔶 PARAMS
QA_DS_NAME, QA_CFG = "squad", None
QUESTION_KEY, CONTEXT_KEY  = "question", "context"
ANS_TEXT_KEY, ANS_START_KEY = "answers.text", "answers.answer_start"
RESUME_FOR_QA = "seqcls_finetuned"          # or None
MAX_LEN, DOC_STRIDE = 384, 128
# 🔶 END

from transformers import AutoModelForQuestionAnswering
qa_ds = load_dataset(QA_DS_NAME, QA_CFG) if not os.path.isdir(QA_DS_NAME) else \
        load_dataset("json", data_files={s:f"{QA_DS_NAME}/{s}.json" for s in ["train","validation"]})

tokenizer = AutoTokenizer.from_pretrained(RESUME_FOR_QA or BASE_MODEL, use_fast=True)
def qa_tok(ex):
    tok = tokenizer(ex[QUESTION_KEY], ex[CONTEXT_KEY],
                    truncation="only_second", max_length=MAX_LEN,
                    stride=DOC_STRIDE, return_overflowing_tokens=True,
                    return_offsets_mapping=True)
    sample_mapping = tok.pop("overflow_to_sample_mapping")
    answers        = ex[ANS_TEXT_KEY]
    starts         = ex[ANS_START_KEY]
    tok["start_positions"], tok["end_positions"] = [], []
    for i, offset in enumerate(tok["offset_mapping"]):
        sample_idx  = sample_mapping[i]
        answer      = answers[sample_idx][0]
        start_char  = starts[sample_idx][0]
        end_char    = start_char + len(answer)
        ctx_offsets = offset[tokenizer.model_input_names.index("offset_mapping"):]
        start_pos = end_pos = 0
        for idx, (s,e) in enumerate(ctx_offsets):
            if s <= start_char < e: start_pos = idx
            if s <  end_char  <= e: end_pos   = idx; break
        tok["start_positions"].append(start_pos)
        tok["end_positions"].append(end_pos)
    tok.pop("offset_mapping")
    return tok

qa_tok_ds = qa_ds.map(qa_tok, batched=True, remove_columns=qa_ds["train"].column_names)
qa_model = AutoModelForQuestionAnswering.from_pretrained(RESUME_FOR_QA or BASE_MODEL).to(device)
qa_args  = TrainingArguments("qa-run", evaluation_strategy="epoch",
                             learning_rate=3e-5, per_device_train_batch_size=8,
                             num_train_epochs=2, fp16=torch.cuda.is_available())
Trainer(qa_model, qa_args,
        train_dataset=qa_tok_ds["train"], eval_dataset=qa_tok_ds["validation"],
        tokenizer=tokenizer).train()
save_ckpt(qa_model, tokenizer, "qa_finetuned")
print("✓ QA model saved at ./qa_finetuned")




# 8 · SEQ2SEQ  — Summarisation or Translation
#     (uses ROUGE or sacreBLEU automatically)

# 🔶 PARAMS
SEQ2_SEQ_MODEL = "t5-small"                 # encoder-decoder checkpoint
SEQ_DS_NAME, SEQ_CFG = "xsum", None         # or "wmt16", "de-en", etc.
DOC_KEY, SUMMARY_KEY = "document", "summary"  # change for translation pair
MAX_SRC, MAX_TGT     = 512, 128
METRIC_NAME          = "rouge"              # or "sacrebleu"
# 🔶 END

from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

seq_ds = load_dataset(SEQ_DS_NAME, SEQ_CFG) if not os.path.isdir(SEQ_DS_NAME) else \
         load_dataset("json", data_files={s:f"{SEQ_DS_NAME}/{s}.json" for s in ["train","validation"]})

tokenizer = AutoTokenizer.from_pretrained(SEQ2_SEQ_MODEL)
def seq_tok(ex):
    model_in  = tokenizer(ex[DOC_KEY], max_length=MAX_SRC, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(ex[SUMMARY_KEY], max_length=MAX_TGT, truncation=True)
    model_in["labels"] = labels["input_ids"]
    return model_in
seq_tok_ds = seq_ds.map(seq_tok, batched=True, remove_columns=seq_ds["train"].column_names)

seq_model = AutoModelForSeq2SeqLM.from_pretrained(SEQ2_SEQ_MODEL).to(device)
seq_coll  = DataCollatorForSeq2Seq(tokenizer, model=seq_model)
seq_args  = Seq2SeqTrainingArguments("seq2seq-run", predict_with_generate=True,
                                     evaluation_strategy="epoch",
                                     learning_rate=2e-5,
                                     per_device_train_batch_size=8,
                                     num_train_epochs=3,
                                     fp16=torch.cuda.is_available())

metric = evaluate.load(METRIC_NAME)
def postprocess_text(preds, labels):
    preds  = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    if METRIC_NAME == "rouge":      # expects newline separating sentences
        preds  = ["\n".join(p.split()) for p in preds]
        labels = ["\n".join(l.split()) for l in labels]
    return preds, labels

def seq_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(eval_pred.label_ids != -100, eval_pred.label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    preds, labels = postprocess_text(decoded_preds, decoded_labels)
    scores = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    if METRIC_NAME == "rouge":
        scores = {k: round(v.mid.fmeasure * 100, 2) for k,v in scores.items()}
    return scores

Seq2SeqTrainer(seq_model, seq_args,
               train_dataset=seq_tok_ds["train"],
               eval_dataset=seq_tok_ds["validation"],
               tokenizer=tokenizer,
               data_collator=seq_coll,
               compute_metrics=seq_metrics).train()
save_ckpt(seq_model, tokenizer, "seq2seq_finetuned")
print("✓ Seq2Seq model saved at ./seq2seq_finetuned")


# 9 · LLM GENERATION  – quick sampler for GPT-style checkpoints
#     (covers the “decoding knobs” part of the camp notebook)

# 🔶 PARAMS
GEN_MODEL   = "gpt2-medium"         # or "meta-llama/Llama-3-8B-Instruct" if RAM allows
PROMPT      = "Explain in two sentences why transformers beat RNNs."   # text prompt
MAX_TOK     = 128
DECODE_KW   = dict(                 # edit combos on the fly
    do_sample=True,                 # False = greedy / beam
    top_k=50,                       # 0 = disabled
    top_p=0.9,
    temperature=1.0,
    num_beams=1,
    repetition_penalty=1.1,
)
USE_LORA    = False                 # flip to True to wrap with LoRA quickly
# 🔶 END

from transformers import AutoModelForCausalLM, TextStreamer
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)  # memory-save
tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL,
                                                 quantization_config=bnb,
                                                 device_map="auto")

if USE_LORA:
    l_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
                       task_type="CAUSAL_LM")
    gen_model = get_peft_model(gen_model, l_cfg).to(device)
    print(gen_model.print_trainable_parameters())

ids = tok(PROMPT, return_tensors="pt").to(gen_model.device)
streamer = TextStreamer(tok)   # prints tokens live
out = gen_model.generate(**ids, max_new_tokens=MAX_TOK,
                         pad_token_id=tok.eos_token_id,
                         streamer=streamer, **DECODE_KW)
# final text
print(tok.decode(out[0], skip_special_tokens=True))



</code></pre>

</details>



