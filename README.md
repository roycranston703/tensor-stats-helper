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
# Section 1 · Weather V1  ―  At-Home Solution (fully annotated)
# “### ADD” marks every line / block that diverges from the organiser baseline.

# Imports
import math, random, datetime as dt
from pathlib import Path, PurePath
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as td
from torch.cuda.amp import autocast, GradScaler

# Config (baseline kept only bands/batch/device)
CFG = dict(
    bands      = 16,
    batch      = 32,
    epochs     = 30,           ### ADD  – baseline had 10
    lr         = 5e-4,         ### ADD  – adamw instead of sgd 1e-3
    device     = "cuda" if torch.cuda.is_available() else "cpu",
    pos_weight = None,         ### ADD  – filled by calibrate_pos_weight()
    focal_alpha= None,         ### ADD  – set alongside pos_weight
    thr        = 0.4           ### ADD  – tuned Dice/Acc threshold
)

# ---------------------- dataset + metadata ----------------------
def sun_elev(lat, lon, utc):
    jd = utc.timetuple().tm_yday + utc.hour/24
    decl = 23.44*math.cos(math.radians((jd+10)*360/365))
    ha   = (utc.hour*15 + lon) - 180
    elev = math.asin(
        math.sin(math.radians(lat))*math.sin(math.radians(decl)) +
        math.cos(math.radians(lat))*math.cos(math.radians(decl))*math.cos(math.radians(ha))
    )
    return math.sin(elev)      # −1 … 1

def load_pt(p: Path):
    t = torch.load(p)          # 17×H×W  (16 bands + mask)
    x, y = t[:-1].float(), t[-1].long()
    # metadata from filename “…_lat13.5_lon102.3_20250815T1710.pt”
    lat = float(PurePath(p).stem.split("_lat")[1].split("_")[0])
    lon = float(PurePath(p).stem.split("_lon")[1].split("_")[0])
    utc = dt.datetime.strptime(PurePath(p).stem.split("_")[-1], "%Y%m%dT%H%M")
    meta = torch.tensor([lat/90, lon/180, sun_elev(lat, lon, utc)], dtype=torch.float32)
    return x, y, meta

class SatDS(td.Dataset):
    def __init__(self, root, split):
        self.files = sorted(Path(root, split).glob("*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):  return load_pt(self.files[i])

def collate(batch):
    xs, ys, ms = zip(*batch)
    xs, ys, ms = torch.stack(xs), torch.stack(ys), torch.stack(ms)
    CFG["img_shape"] = xs.shape[-2:]
    return xs, ys, ms

def make_loader(root, split):
    return td.DataLoader(SatDS(root, split),
                         batch_size=CFG["batch"],
                         shuffle=(split=="train"),
                         collate_fn=collate)

# -------------------------- augmentation ------------------------
def random_band_drop(x, p=0.2):          ### ADD
    if random.random() < p:
        x[:, torch.randint(0, x.size(1), ())] = 0
    return x
def add_noise(x, σ=0.01): return x + σ*torch.randn_like(x)   ### ADD
def hor_flip(x, y):
    if random.random() < 0.5:
        x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
    return x, y
def aug(x, y):
    x = random_band_drop(x); x = add_noise(x)
    x, y = hor_flip(x, y)
    return x, y

# ------------------------ model ---------------------------------
class ResBlock(nn.Module):               ### ADD (InstanceNorm + 2d-Dropout)
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.InstanceNorm2d(c); self.n2 = nn.InstanceNorm2d(c)
        self.c1 = nn.Conv2d(c,c,3,1,1); self.c2 = nn.Conv2d(c,c,3,1,1)
        self.drop, self.act = nn.Dropout2d(0.1), nn.SiLU()
    def forward(self, x):
        h = self.act(self.n1(x)); h = self.drop(self.c1(h))
        h = self.act(self.n2(h)); h = self.c2(h)
        return x + h

class FiLM(nn.Module):                   ### ADD – scalar conditioning
    def __init__(self, ch, cond=3, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond,hid), nn.ReLU(),
                                 nn.Linear(hid,ch*2))
    def forward(self, f, meta):
        γβ = self.mlp(meta); γ, β = γβ.chunk(2,1)
        return f*(1+γ.view(-1,f.size(1),1,1)) + β.view(-1,f.size(1),1,1)

class UNetV1(nn.Module):
    def __init__(self, in_ch=16, use_attn=False):   ### use_attn reserved for CBAM
        super().__init__()
        self.stem = nn.Conv2d(in_ch,64,3,1,1)
        self.d1 = nn.Conv2d(64,128,4,2,1); self.rb1=ResBlock(128)
        self.d2 = nn.Conv2d(128,256,4,2,1); self.rb2=ResBlock(256)
        self.d3 = nn.Conv2d(256,512,4,2,1); self.rb3=ResBlock(512)
        self.mid = ResBlock(512)
        self.film= FiLM(512,3)
        self.u3 = nn.ConvTranspose2d(512,256,4,2,1)
        self.u2 = nn.ConvTranspose2d(512,128,4,2,1)
        self.u1 = nn.ConvTranspose2d(256,64,4,2,1)
        self.out= nn.Conv2d(128,1,1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(), nn.Linear(512,1))
    def forward(self, x, meta):
        s  = self.stem(x)
        d1 = self.rb1(self.d1(s))
        d2 = self.rb2(self.d2(d1))
        d3 = self.rb3(self.d3(d2))
        m  = self.film(self.mid(d3), meta)
        u3 = self.u3(m)
        u2 = self.u2(torch.cat([u3,d2],1))
        u1 = self.u1(torch.cat([u2,d1],1))
        mask = self.out(torch.cat([u1,s],1))
        flag = self.cls(m).squeeze(1)
        return mask, flag

# -------------------- dynamic class weight + focal α ------------
def calibrate_pos_weight(loader_tr):     ### ADD
    fg, px = 0, 0
    for _, y, _ in loader_tr:
        fg += y.sum().item(); px += y.numel()
    p = fg / px
    CFG["pos_weight"] = (1-p)/p
    CFG["focal_alpha"]= 1 - p            # FG weight in focal loss
    print(f"class-imbalance p={p:.3%}  pos_w={CFG['pos_weight']:.4f}  α={CFG['focal_alpha']:.4f}")

# --------------------------- losses ------------------------------
class DiceLoss(nn.Module):
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        inter = (p*y).sum(); union = p.sum()+y.sum()
        return 1 - (2*inter+1)/(union+1)

class FocalLoss(nn.Module):             ### ADD – uses calibrated α
    def __init__(self, γ=2):
        super().__init__(); self.γ=γ
    def forward(self, logit, y):
        α = CFG["focal_alpha"]
        p  = torch.sigmoid(logit)
        pt = p*y + (1-p)*(1-y)
        w  = α*y + (1-α)*(1-y)
        return (w*((1-pt)**self.γ)*(-pt.log())).mean()

def active_contour(logit, y, λ=1, μ=1):
    p = torch.sigmoid(logit)
    dy, dx = torch.gradient(p, dim=(2,3))
    length = torch.sqrt((dx**2+dy**2)+1e-8).mean()
    region = (λ*((p-y)**2)*y + μ*((p-y)**2)*(1-y)).mean()
    return length+region

class WeatherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(); self.focal = FocalLoss()
    def forward(self, mask_logit, img_logit, y):
        flag = (y.sum((1,2))>0).float()
        seg = 0.5*self.focal(mask_logit,y) + 0.3*self.dice(mask_logit,y) + \
              0.2*active_contour(mask_logit,y)
        img = F.binary_cross_entropy_with_logits(img_logit, flag)
        return seg + 0.2*img

# ------------------------ metric -------------------------------
def dice_acc(mask_logit, img_logit, y, thr=CFG["thr"]):
    dice = 1 - DiceLoss()(mask_logit, y)
    pred_flag = (torch.sigmoid(img_logit)>0.5).long()
    acc = (pred_flag == (y.sum((1,2))>0)).float().mean()
    return 0.5*(dice+acc)

# ------------------------ training -----------------------------
def train(root):
    tr = make_loader(root,"train")
    calibrate_pos_weight(tr)            # sets pos_weight & α
    va = make_loader(root,"val")
    net=UNetV1(CFG["bands"]).to(CFG["device"])
    opt=torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=1e-2)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"],
                           total_steps=len(tr)*CFG["epochs"])
    scaler, criterion = GradScaler(), WeatherLoss()
    best = 0
    for ep in range(CFG["epochs"]):
        net.train()
        for x,y,m in tr:
            x,y,m = x.to(CFG["device"]),y.to(CFG["device"]),m.to(CFG["device"])
            x,y = aug(x,y)
            with autocast():
                mask,flag = net(x,m)
                loss = criterion(mask,flag,y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        net.eval(); vals=[]
        with torch.no_grad(), autocast():
            for x,y,m in va:
                x,y,m = x.to(CFG["device"]),y.to(CFG["device"]),m.to(CFG["device"])
                mask,flag = net(x,m)
                vals.append(dice_acc(mask,flag,y))
        val = torch.tensor(vals).mean().item()
        if val>best: best=val; torch.save(net.state_dict(),"weather_best.pth")
        print(f"ep{ep:02d} val {val:.4f}")

# call training:
# train("/path/to/weather_dataset")

# Section 2 · Config + Loader + Meta-parser
# This chunk covers S-0 and S-1 in one go.
# Drop it at the top of your satellite notebook.

import math, datetime as dt
from pathlib import Path, PurePath
import torch, torch.utils.data as td

# ---------- CFG – edit once, everything downstream picks it up ----------
CFG = dict(
    # data-specific
    bands       = 16,          # change if organiser adds/removes channels
    img_shape   = None,        # auto-filled after first batch
    use_meta    = True,        # lat/lon/UTC → sun-elev conditioning
    metric_name = "dice_acc",  # default competition metric
    # training hyper-params (overridden later if needed)
    batch       = 32,
    epochs      = 30,
    lr          = 5e-4,
    device      = "cuda" if torch.cuda.is_available() else "cpu",
    # class-imbalance weights (filled by calibrate_pos_weight later)
    pos_weight  = None,
    focal_alpha = None,
    thr         = 0.4          # initial mask threshold for Dice/Acc blend
)

# ---------- tiny helper: sun elevation normalised to −1 … 1 ----------
def sun_elev(lat, lon, utc):
    jd   = utc.timetuple().tm_yday + utc.hour / 24
    decl = 23.44 * math.cos(math.radians((jd + 10) * 360 / 365))
    ha   = (utc.hour * 15 + lon) - 180          # hour angle
    elev = math.asin(
        math.sin(math.radians(lat)) * math.sin(math.radians(decl)) +
        math.cos(math.radians(lat)) * math.cos(math.radians(decl)) *
        math.cos(math.radians(ha))
    )
    return math.sin(elev)                       # −1…1

# ---------- file → (x, y, meta)  loader ----------------------------------
def load_pt(f: Path):
    """
    Expects organiser .pt with 17×H×W tensor:
        16 bands (float32) + 1 binary mask (int64)
    Filename carries metadata:
        “…_lat13.5_lon102.3_20250815T1710.pt”
    Returns:
        x : 16×H×W  float32   (NaNs replaced by 0)
        y :    H×W  int64
        m : 3-dim meta tensor  (lat, lon, sun-elev)
    """
    t   = torch.load(f)                         # shape 17×H×W
    x   = t[:-1].float()
    x[torch.isnan(x)] = 0                       # NaNs → 0  (handles band gaps)
    y   = t[-1].long()

    if CFG["use_meta"]:
        lat = float(PurePath(f).stem.split("_lat")[1].split("_")[0])
        lon = float(PurePath(f).stem.split("_lon")[1].split("_")[0])
        utc = dt.datetime.strptime(PurePath(f).stem.split("_")[-1], "%Y%m%dT%H%M")
        meta = torch.tensor([lat / 90, lon / 180, sun_elev(lat, lon, utc)],
                            dtype=torch.float32)
    else:
        meta = torch.zeros(3)

    return x, y, meta

# ---------- torch Dataset / DataLoader -----------------------------------
class SatDataset(td.Dataset):
    def __init__(self, root, split):
        self.files = sorted(Path(root, split).glob("*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return load_pt(self.files[idx])

def collate_fn(batch):
    xs, ys, ms = zip(*batch)
    xs, ys, ms = torch.stack(xs), torch.stack(ys), torch.stack(ms)
    # remember true image shape for dynamic padding later
    CFG["img_shape"] = xs.shape[-2:]
    return xs, ys, ms

def make_loader(root, split):
    """
    root/
      └── train/*.pt
          val/*.pt
    """
    return td.DataLoader(SatDataset(root, split),
                         batch_size=CFG["batch"],
                         shuffle=(split == "train"),
                         collate_fn=collate_fn)
# Section 3 · Augmentation Bag (Spectral & Geometric)
# Hook into training with:   x, y = apply_sat_aug(x, y)
# Each aug is a plain function so you can reorder or comment out lines.

import torch, random, torch.nn.functional as F

# -------- band-level corruption ---------------------------------
def random_band_drop(x, p=0.2):
    """
    Zero-out one spectral band with prob p.
    Guards against real validation files where a VIS/IR channel is missing.
    """
    if random.random() < p:
        band = torch.randint(0, x.size(1), ())
        x[:, band] = 0
    return x

def gaussian_noise(x, sigma=0.01):
    """
    Additive Gaussian noise – covers sensor SNR shifts or compression artefacts.
    """
    return x + sigma * torch.randn_like(x)

# -------- sample-level blending ---------------------------------
def mixup(x, y, alpha=0.4):
    """
    MixUp for segmentation: convex combination of two images.
    Only use when your loss can handle soft labels (e.g. BCE/Dice).
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    # keep hard label of dominant sample for simplicity
    y_mix = y if lam >= 0.5 else y[idx]
    return x_mix, y_mix

def cutmix_patch(x, y, alpha=1.0, max_prop=0.4):
    """
    CutMix – paste random patch from another image in the batch.
    More aggressive than MixUp; good against over-fitting when train < val size.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()
    cut_w = int(W * max_prop * lam ** 0.5)
    cut_h = int(H * max_prop * lam ** 0.5)
    cx, cy = torch.randint(0, W, (1,)), torch.randint(0, H, (1,))
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    idx = torch.randperm(B)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    y[:,   y1:y2, x1:x2] = y[idx,   y1:y2, x1:x2]
    return x, y

# -------- geometric invariance ---------------------------------
def horizontal_flip(x, y):
    """
    Flip along longitude – valid because physical latitude order stays.
    Disable if organiser’s task attaches absolute longitudes to classes!
    """
    if random.random() < 0.5:
        x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
    return x, y

# -------- master switchboard -----------------------------------
def apply_sat_aug(x, y, *, use_mixup=False, use_cutmix=False):
    """
    Call inside training loop *before* sending to model.
      x, y = apply_sat_aug(x, y)
    Toggle MixUp / CutMix via kwargs.
    """
    x = random_band_drop(x)
    x = gaussian_noise(x)
    x, y = horizontal_flip(x, y)
    if use_mixup:
        x, y = mixup(x, y)
    if use_cutmix:
        x, y = cutmix_patch(x, y)
    return x, y
# Section 4 · Metadata Conditioning Blocks  (S-3)
# Plug the returned module into your UNet bottleneck:
#
#     self.meta_mod = build_meta_block(feat_ch=512,
#                                      mode="film",      # or "channel"
#                                      cond_dim=3)       # length of meta vector
#     ...
#     feats = self.meta_mod(feats, meta_vec)
#
# Modes
#   "film"     – FiLM γ/β modulation   (default, good for small scalar vectors)
#   "channel"  – Channel attention     (sigmoid weights)   use when meta vector
#                should softly gate each feature map.

import torch, torch.nn as nn

# ---------- core primitives ------------------------------------
class FiLM(nn.Module):
    """Feature-wise linear modulation: feats * (1+γ) + β."""
    def __init__(self, feat_ch, cond_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_ch * 2)
        )
    def forward(self, feats, meta):
        γβ = self.net(meta)               # B×2C
        γ, β = γβ.chunk(2, dim=1)
        γ = γ.view(-1, feats.size(1), 1, 1)
        β = β.view(-1, feats.size(1), 1, 1)
        return feats * (1 + γ) + β

class ChannelAttention(nn.Module):
    """Apply sigmoid gate per feature map: feats * σ(Meta→C)."""
    def __init__(self, feat_ch, cond_dim, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_ch),  nn.Sigmoid()
        )
    def forward(self, feats, meta):
        g = self.fc(meta).view(-1, feats.size(1), 1, 1)
        return feats * g

# ---------- factory helper ------------------------------------
def build_meta_block(feat_ch, mode="film", cond_dim=3, hidden=64):
    """
    mode : "film" | "channel" | None
    cond_dim : length of meta vector (e.g. 3 if [lat, lon, sun])
    """
    if mode is None or cond_dim == 0:
        return nn.Identity()
    if mode == "film":
        return FiLM(feat_ch, cond_dim, hidden)
    if mode == "channel":
        return ChannelAttention(feat_ch, cond_dim, hidden)
    raise ValueError(f"Unknown meta conditioning mode: {mode}")
# How to use Inside your UNet bottleneck
self.meta_mod = build_meta_block(feat_ch=512,
                                 mode="film",      # or "channel", or None
                                 cond_dim=meta.size(1))

...
feats = self.meta_mod(feats, meta)   # meta is B×cond_dim

# Section 5 · Backbone — Residual UNet (InstanceNorm, Dropout2d, optional CBAM)
# This is the *conventional, explicit* layer-by-layer version —
# no loops, no dynamic padding, mirrors the style you used in Weather V1.

import torch, torch.nn as nn

# ------------------------------------------------------------
# Optional CBAM attention  (channel- & spatial)
# ------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, ch, red=16, k=7):
        super().__init__()
        self.mlp  = nn.Sequential(nn.Linear(ch, ch // red), nn.ReLU(),
                                  nn.Linear(ch // red, ch))
        self.conv = nn.Conv2d(2, 1, k, padding=(k - 1) // 2)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        att = self.mlp(x.mean((2, 3)).view(b, c)) + \
              self.mlp(x.amax((2, 3)).view(b, c))
        x   = x * self.sig(att).view(b, c, 1, 1)
        spa = self.sig(self.conv(torch.cat([x.mean(1, True),
                                            x.amax(1, True)], 1)))
        return x * spa

# ------------------------------------------------------------
# Residual Conv Block:   IN → CONV → Drop2d → IN → CONV (+ res) → CBAM?
# ------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dropout_p=0.1, use_attn=False):
        super().__init__()
        self.n1 = nn.InstanceNorm2d(ch)
        self.n2 = nn.InstanceNorm2d(ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.drop, self.act = nn.Dropout2d(dropout_p), nn.SiLU()
        self.attn = CBAM(ch) if use_attn else nn.Identity()

    def forward(self, x):
        h = self.act(self.n1(x))
        h = self.drop(self.c1(h))
        h = self.act(self.n2(h))
        h = self.c2(h)
        return self.attn(x + h)

# ------------------------------------------------------------
# UNet-Res backbone, depth = 4 (handles up to 256 × 256 cleanly)
#   • meta_block: any nn.Module(feats, meta)  (FiLM, channel-attn …)
# ------------------------------------------------------------
class UNetRes(nn.Module):
    """
    Args
    ----
    in_ch      : # input spectral bands (e.g. 16)
    base       : # filters after stem (64 default)
    dropout_p  : 2-D dropout prob inside every ResBlock
    use_attn   : True → wrap each ResBlock with CBAM
    meta_block : optional conditioning module; Identity if None
    """
    def __init__(self, in_ch, base=64,
                 dropout_p=0.1, use_attn=False,
                 meta_block=None):
        super().__init__()
        # Stem
        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)

        # Encoder
        self.down1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.rb1   = ResBlock(base*2, dropout_p, use_attn)

        self.down2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.rb2   = ResBlock(base*4, dropout_p, use_attn)

        self.down3 = nn.Conv2d(base*4, base*8, 4, 2, 1)
        self.rb3   = ResBlock(base*8, dropout_p, use_attn)

        # Bottleneck
        self.mid   = ResBlock(base*8, dropout_p, use_attn)
        self.meta  = meta_block if meta_block is not None else nn.Identity()

        # Decoder
        self.up3   = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1)
        self.cv3   = nn.Conv2d(base*8, base*4, 3, 1, 1)

        self.up2   = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1)
        self.cv2   = nn.Conv2d(base*4, base*2, 3, 1, 1)

        self.up1   = nn.ConvTranspose2d(base*2, base,   4, 2, 1)
        self.cv1   = nn.Conv2d(base*2, base,   3, 1, 1)

        self.out_channels = base     # expose for head attachment
        self.tail = nn.Conv2d(base*2, base, 3, 1, 1)   # concat stem skip later

    def forward(self, x, meta=None):
        s0 = self.stem(x)            # B × base × H × W

        d1 = self.rb1(self.down1(s0))  # B × 2B × H/2
        d2 = self.rb2(self.down2(d1))  # B × 4B × H/4
        d3 = self.rb3(self.down3(d2))  # B × 8B × H/8

        bott = self.meta(self.mid(d3), meta)            # apply FiLM if any

        u3 = self.up3(bott)                             # H/4
        u3 = self.cv3(torch.cat([u3, d2], 1))

        u2 = self.up2(u3)                               # H/2
        u2 = self.cv2(torch.cat([u2, d1], 1))

        u1 = self.up1(u2)                               # H
        u1 = self.cv1(torch.cat([u1, s0], 1))

        feats = self.tail(torch.cat([u1, s0], 1))       # final feature map
        return feats
# Section 6 · Heads (S-5)
# Attach exactly one of these heads to the backbone’s final feature map.
#
# Usage example
# -------------
#     feats = backbone(x, meta)           # B × C × H × W
#     head   = build_head(feat_ch=backbone.out_channels,
#                         head_type="binary",    # binary | multi | reg
#                         num_classes=3)         # used for head_type="multi"
#     logits = head(feats)
#
# Available head types
#   • "binary"   – 1-channel sigmoid mask  (+ optional image-flag)
#   • "multi"    – N-channel softmax mask
#   • "reg"      – 1-channel rain-rate regression (mm h⁻¹)
# If you need both pixel mask *and* image-flag, set `include_flag=True`.

import torch, torch.nn as nn

# ---------------- basic pixel heads ------------------------------------
class BinaryMaskHead(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, 1, 1)
    def forward(self, feats):  return self.conv(feats)          # logits

class MultiClassHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)
    def forward(self, feats):  return self.conv(feats)          # logits

class RegressionHead(nn.Module):
    def __init__(self, in_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, 1, 1)
    def forward(self, feats):  return self.conv(feats)          # linear value

# ---------------- optional image-level flag ----------------------------
class ImageFlag(nn.Module):
    """Global rain / no-rain flag via GAP + FC."""
    def __init__(self, in_ch):
        super().__init__()
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 1)
        )
    def forward(self, feats):  return self.cls(feats).squeeze(1)

# ---------------- factory helper --------------------------------------
class HeadWrapper(nn.Module):
    """
    Returns a tuple:   (pixel_output, img_flag or None)
    pixel_output shape:
        binary, reg    → B×1×H×W
        multi          → B×num_classes×H×W
    """
    def __init__(self, feat_ch, head_type="binary", num_classes=2,
                 include_flag=False):
        super().__init__()
        if head_type == "binary":
            self.pix = BinaryMaskHead(feat_ch)
        elif head_type == "multi":
            self.pix = MultiClassHead(feat_ch, num_classes)
        elif head_type == "reg":
            self.pix = RegressionHead(feat_ch)
        else:
            raise ValueError("head_type must be binary | multi | reg")

        self.flag = ImageFlag(feat_ch) if include_flag else None

    def forward(self, feats):
        pixel = self.pix(feats)
        fflag = self.flag(feats) if self.flag else None
        return pixel, fflag

def build_head(feat_ch, head_type="binary", num_classes=2, include_flag=True):
    """
    Convenience wrapper:
        head = build_head(backbone.out_channels, "binary", include_flag=True)
    """
    return HeadWrapper(feat_ch, head_type, num_classes, include_flag)
# Section 7 · Loss Bank & Mixer  (S-6)
# Each loss takes logits + ground-truth mask (plus img_flag when needed).
# Combine any subset with one line:
#
#   criterion = CombinedLoss(
#       names   = ["focal", "dice", "contour", "flag_bce"],
#       weights = [0.4,      0.3,    0.1,      0.2]   # auto-normalised
#   )
#
# Dynamic foreground weight & focal α are read from CFG, set once by
# `calibrate_pos_weight(loader_tr)` (see Section 2).

import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- individual pixel losses ---------------------------------
class DiceLoss(nn.Module):
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        inter = (p*y).sum(); union = p.sum()+y.sum()
        return 1 - (2*inter+1)/(union+1)

class FocalLoss(nn.Module):
    def __init__(self, γ=2):
        super().__init__(); self.γ = γ
    def forward(self, logit, y):
        α = CFG["focal_alpha"]         # set by calibrate_pos_weight
        p  = torch.sigmoid(logit)
        pt = p*y + (1-p)*(1-y)
        w  = α*y + (1-α)*(1-y)
        return (w * (1-pt).pow(self.γ) * (-pt.log())).mean()

def active_contour(logit, y, λ=1, μ=1):
    p = torch.sigmoid(logit)
    dy, dx = torch.gradient(p, dim=(2,3))
    length = torch.sqrt((dx**2 + dy**2) + 1e-8).mean()
    region = (λ*((p-y)**2)*y + μ*((p-y)**2)*(1-y)).mean()
    return length + region

class TverskyLoss(nn.Module):
    def __init__(self, α=0.7, β=0.3):
        super().__init__(); self.a, self.b = α, β
    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        tp = (p*y).sum(); fp = (p*(1-y)).sum(); fn = ((1-p)*y).sum()
        return 1 - (tp + 1) / (tp + self.a*fp + self.b*fn + 1)

# Lovász hinge surrogate for IoU (binary)
def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - inter / union
    jaccard[1:] -= jaccard[:-1]
    return jaccard

def lovasz_binary_flat(logits, labels):
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)

class LovaszHinge(nn.Module):
    def forward(self, logit, y):
        return lovasz_binary_flat(logit.view(-1), y.view(-1))

# image-level flag BCE
class FlagBCE(nn.Module):
    def forward(self, img_logit, y_mask):
        flag = (y_mask.sum((1,2)) > 0).float()
        return F.binary_cross_entropy_with_logits(img_logit, flag)

# ---------------- registry -----------------------------------------------
LOSS_BANK = {
    "focal"    : FocalLoss,
    "dice"     : DiceLoss,
    "contour"  : lambda: active_contour,    # functional form
    "tversky"  : TverskyLoss,
    "lovasz"   : LovaszHinge,
    "flag_bce" : FlagBCE
}

# ---------------- mixer ---------------------------------------------------
class CombinedLoss(nn.Module):
    """
    names   : list of keys from LOSS_BANK.
    weights : same length; will be re-normalised.
    Example:
        criterion = CombinedLoss(["focal","dice","flag_bce"],
                                 [0.5,    0.3,   0.2])
    """
    def __init__(self, names, weights):
        super().__init__()
        assert len(names) == len(weights) and all(n in LOSS_BANK for n in names)
        # convert weights → tensor & normalise
        w = torch.tensor(weights, dtype=torch.float)
        self.weights = (w / w.sum()).tolist()
        # instantiate or keep callable
        self.loss_fns = []
        for n in names:
            lf = LOSS_BANK[n]()
            self.loss_fns.append(lf)

    def forward(self, mask_logit, img_logit, y):
        total = 0.
        for w, fn in zip(self.weights, self.loss_fns):
            if isinstance(fn, nn.Module):
                # pixel-wise loss
                loss = fn(mask_logit, y) if not isinstance(fn, FlagBCE) \
                       else fn(img_logit, y)
            else:
                # functional contour loss
                loss = fn(mask_logit, y)
            total += w * loss
        return total
# Classic V1 mix  (Focal + Dice + Contour + Flag)
crit = CombinedLoss(["focal","dice","contour","flag_bce"],
                    [0.4,   0.3,  0.1,      0.2])

# Metric flips to IoU only
crit = CombinedLoss(["lovasz"], [1.0])

# Heavy FP penalty scenario
crit = CombinedLoss(["tversky","flag_bce"], [0.8, 0.2])
# Section 8 · Training Engine  (build → train → validate → save best)
# One entry-point:    train_sat(root_dir)
# Flags live in CFG (optim, sched, epochs, grad_clip, etc.) so you flip
# behaviour without rewriting the loop.

import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (OneCycleLR, CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)

# ------------ optimiser / scheduler builders -----------------------------
def build_optimizer(model):
    lr = CFG["lr"]; wd = 1e-2
    return optim.AdamW(model.parameters(), lr, weight_decay=wd)  # good default

def build_scheduler(opt, steps_per_epoch):
    return OneCycleLR(opt, max_lr=CFG["lr"],
                      total_steps=steps_per_epoch * CFG["epochs"])

# ------------ training loop ----------------------------------------------
def train_sat(root):
    # 1) loaders
    loader_tr = make_loader(root, "train")        # from Section 2
    calibrate_pos_weight(loader_tr)               # sets pos_weight & focal α
    loader_va = make_loader(root, "val")

    # 2) build model = backbone + head
    meta_mod  = build_meta_block(512, mode="film", cond_dim=3)           # Sec 4
    backbone  = UNetRes(in_ch=CFG["bands"],
                        dropout_p=0.1,
                        use_attn=True,
                        meta_block=meta_mod)                             # Sec 5
    head      = build_head(backbone.out_channels,
                           head_type="binary",
                           include_flag=True)                            # Sec 6
    model     = nn.Sequential(backbone, head)                            # simple wrap

    model.to(CFG["device"])

    # 3) loss, optim, sched, AMP scaler
    criterion = CombinedLoss(["focal", "dice", "contour", "flag_bce"],
                             [0.4,    0.3,   0.1,      0.2])             # Sec 6
    opt    = build_optimizer(model)
    sched  = build_scheduler(opt, len(loader_tr))
    scaler = GradScaler()
    best   = -1

    # 4) training epochs
    for ep in range(CFG["epochs"]):
        model.train()
        for x, y, meta in loader_tr:
            x, y, meta = x.to(CFG["device"]), y.to(CFG["device"]), meta.to(CFG["device"])
            x, y = apply_sat_aug(x, y)                                   # Sec 3

            with autocast():
                feats   = backbone(x, meta)
                mask_lp, flag_lp = head(feats)
                loss = criterion(mask_lp, flag_lp, y)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

        # ----- validation -----
        val_score = evaluate(model, loader_va)                           # Sec 7
        print(f"ep{ep:02d}  val {val_score:.4f}")

        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), "sat_best.pth")

    # 5) sweep optimal threshold for Dice / IoU blends
    sweep_threshold(model, loader_va, metric_name=CFG["metric_name"])    # Sec 7
    print("Training done. Best val =", best, "  Model saved to sat_best.pth")

# Example run:
# train_sat("/path/to/satellite_dataset")
# Section 9 · Inference + Test-Time Augmentation (S-9)
# ----------------------------------------------------
# One entry point ―  run_inference(test_root)
#   • Loads sat_best.pth + best_thr.json
#   • Optional horizontal flip TTA
#   • Writes masks to test_root/preds/*.pt   1-byte per pixel (uint8)
#
# Adjust CFG["use_tta"] to False if you need speed.

import json, torch, torch.nn.functional as F
from pathlib import Path

# -------- configuration flag --------
CFG["use_tta"] = True      # set False to disable flip-ensemble

# -------- rebuild model exactly as training --------
def load_model():
    meta_mod  = build_meta_block(512, mode="film", cond_dim=3)
    backbone  = UNetRes(in_ch=CFG["bands"],
                        dropout_p=0.1,
                        use_attn=True,
                        meta_block=meta_mod)
    head      = build_head(backbone.out_channels,
                           head_type="binary",
                           include_flag=True)
    model = nn.Sequential(backbone, head).to(CFG["device"])
    model.load_state_dict(torch.load("sat_best.pth", map_location=CFG["device"]))
    model.eval()
    return model

# -------- helper: flip TTA --------
def _forward_tta(model, x, meta):
    m1, f1 = model(x, meta)
    if not CFG["use_tta"]: return m1, f1
    x_flip = torch.flip(x, [-1])
    m2, f2 = model(x_flip, meta)
    m2 = torch.flip(m2, [-1])          # unflip
    m = (m1 + m2) / 2
    f = (f1 + f2) / 2
    return m, f

# -------- load threshold --------
try:
    _THR = json.load(open("best_thr.json"))["thr"]
except FileNotFoundError:
    _THR = CFG["thr"]

# -------- main inference routine --------
@torch.no_grad()
def run_inference(root):
    test_loader = make_loader(root, "test")      # uses collate_fn Section 2
    model = load_model()

    out_dir = Path(root, "preds"); out_dir.mkdir(exist_ok=True)
    for idx, (x, _, meta) in enumerate(test_loader):
        x, meta = x.to(CFG["device"]), meta.to(CFG["device"])
        m_logit, _ = _forward_tta(model, x, meta)
        masks = (torch.sigmoid(m_logit) > _THR).byte().cpu()   # uint8 0/1
        for i, mask in enumerate(masks):
            # save one file per sample   e.g. preds/idx_00012.pt
            torch.save(mask.squeeze(0), out_dir / f"idx_{idx*CFG['batch']+i:05d}.pt")
    print("Inference done. Masks saved to", out_dir)
# Section 10 · Few-Shot Adapt Helper  (S-10)
# -----------------------------------------------------------
# Quickly fine-tune sat_best.pth on a tiny organiser-supplied
# adaptation set (e.g. 10–50 images) and save adapted.pth.
#
# Key knobs
#   • freeze_encoder : True → only decoder + head learn
#   • epochs         : default 5  (fast)
#   • lr             : default 3e-4 (lower than full training)
#
# Usage
#   adapt_fewshot(adapt_root="adat_set",
#                 base_ckpt="sat_best.pth",
#                 out_ckpt="adapted.pth",
#                 freeze_encoder=True)

def adapt_fewshot(adapt_root,
                  base_ckpt="sat_best.pth",
                  out_ckpt="adapted.pth",
                  freeze_encoder=True,
                  epochs=5,
                  lr=3e-4):

    # ---------- loaders (reuse Section 2 make_loader) ----------
    tr = make_loader(adapt_root, "train")
    va = make_loader(adapt_root, "val")

    # ---------- rebuild model & load base weights -------------
    meta_mod = build_meta_block(512, mode="film", cond_dim=3)
    backbone = UNetRes(in_ch=CFG["bands"],
                       dropout_p=0.1,
                       use_attn=True,
                       meta_block=meta_mod)
    head     = build_head(backbone.out_channels,
                          head_type="binary",
                          include_flag=True)
    model = nn.Sequential(backbone, head).to(CFG["device"])
    model.load_state_dict(torch.load(base_ckpt, map_location=CFG["device"]))

    # freeze encoder if requested
    if freeze_encoder:
        for n, p in model.named_parameters():
            if "down" in n or "stem" in n or "mid" in n:
                p.requires_grad_(False)

    # ---------- optimiser / sched ----------
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=len(tr)*epochs)
    scaler, criterion = GradScaler(), CombinedLoss(
        ["focal","dice","flag_bce"], [0.5,0.3,0.2])

    best = -1
    for ep in range(epochs):
        model.train()
        for x, y, meta in tr:
            x,y,meta = x.to(CFG["device"]),y.to(CFG["device"]),meta.to(CFG["device"])
            x,y = apply_sat_aug(x,y, use_mixup=False, use_cutmix=False)  # light aug
            with autocast():
                feats = backbone(x, meta)
                m_log, f_log = head(feats)
                loss = criterion(m_log, f_log, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

        # val metric
        val = evaluate(model, va)
        print(f"[adapt] ep{ep}  val {val:.4f}")
        if val > best:
            best = val; torch.save(model.state_dict(), out_ckpt)

    print("Few-shot adaptation done ➜", out_ckpt, "  best val =", best)

</code></pre>

</details>
<details>
<summary>Contributors (click to expand)</summary>

<pre><code class="language-python">
# ================================================================
# SECTION 1 · Text-Only Baseline (organiser MiniLM-L6 + fine-tune)
# ---------------------------------------------------------------
#  • load_icon_db()            → {id: description}
#  • encode_choices()          caches choice vectors
#  • guess_words(hints, opts)  returns top-10 prediction list
#  • fine_tune_20()            one-pass cosine-loss fine-tune on 20 val rounds
# ================================================================

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util
import torch, json, math, random
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMB_DIM    = 384
TOP_K      = 10

# ---------- 0.  icon DB ----------
def load_icon_db(path="icon_descriptions.json"):
    return json.loads(Path(path).read_text())

ICON_DB = load_icon_db()            # {id: "A red apple …"}

# ---------- 1.  encoder + cache ----------
text_encoder = SentenceTransformer(BASE_MODEL, device=device)
_choice_cache = {}
def encode_choices(choices):
    miss = [c for c in choices if c not in _choice_cache]
    if miss:
        vecs = text_encoder.encode([f"A {x}" for x in miss],
                                   convert_to_tensor=True, show_progress_bar=False)
        for k,v in zip(miss, vecs):
            _choice_cache[k] = v / v.norm()
    return torch.stack([_choice_cache[c] for c in choices]).to(device)   # (N,384)

# ---------- 2.  hint prompt builder ----------
def hints_to_sentence(hints):
    # preserves order; works for 1–5 hints
    return " -> ".join([ICON_DB[h]['description'].lower() for h in hints])

# ---------- 3.  main guesser ----------
def guess_words(hints: list[int], choices: list[str]) -> list[str]:
    q   = hints_to_sentence(hints)
    qv  = text_encoder.encode(q, convert_to_tensor=True).to(device)
    qv  = qv / qv.norm()
    cv  = encode_choices(choices)                     # (N,384)
    sims= (qv @ cv.T).cpu()                           # (N,)
    top = sims.topk(TOP_K).indices
    return [choices[i] for i in top]

# ---------- 4.  optional fine-tune on 20 validation rounds ----------
def fine_tune_20(val_path="takehome_validation.json"):
    data  = json.loads(Path(val_path).read_text())
    rand  = random.Random(42)
    train = []
    for row in data:
        hints = [h for h in row['hints'] if h in ICON_DB]
        sent  = hints_to_sentence(hints)
        pos   = row['label']
        neg   = rand.choice([c for c in row['options'] if c != pos])
        train.append(InputExample(texts=[sent, f"A {pos}"], label=1.0))
        train.append(InputExample(texts=[sent, f"A {neg}"], label=0.0))
    ds   = SentencesDataset(train, text_encoder)
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=8)
    loss   = losses.CosineSimilarityLoss(text_encoder)
    text_encoder.fit(train_objectives=[(loader, loss)],
                     epochs=1, warmup_steps=10)
    print("⇒ Mini fine-tune done.")
# ================================================================
# SECTION 2 · CLIP Fusion (icons + descriptions) — Weather-team V2
# ---------------------------------------------------------------
#  • build_dataloaders()    loads 64×64 icon PNG + description
#  • train_clip_contrast() fine-tunes ViT-B/32 for 30 epochs
#  • clip_guess(hints, opts, α=0.5)  ranks by α·image+ (1-α)·text
# ================================================================

import os, torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # mainland mirror
device   = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_name= 'openai/clip-vit-base-patch16'
clipM    = CLIPModel.from_pretrained(clip_name).to(device)
proc     = CLIPProcessor.from_pretrained(clip_name)

# ---------- 1.  dataset ----------
class IconSet(torch.utils.data.Dataset):
    def __init__(self, icon_db):
        self.ids  = sorted(icon_db)
        self.desc = [f"an icon showing {icon_db[i]['description'].replace('\n',' and ')}"
                     for i in self.ids]
        self.imgs = [icon_db[i]['icons'] for i in self.ids]  # PIL 64×64
    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.imgs[i], self.desc[i]

def build_dataloaders(batch=32):
    ds = IconSet(ICON_DB)
    def collate(batch):
        imgs, txts = zip(*batch)
        enc = proc(images=list(imgs), text=list(txts), return_tensors='pt',
                   padding=True, truncation=True)
        return {k: v.to(device) for k,v in enc.items()}
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       collate_fn=collate)

# ---------- 2.  contrastive fine-tune ----------
def train_clip_contrast(epochs=30, lr=5e-5):
    clipM.train(); opt = torch.optim.AdamW(clipM.parameters(), lr=lr)
    loader = build_dataloaders()
    for ep in range(epochs):
        tot = 0
        for batch in loader:
            opt.zero_grad()
            out = clipM(**batch)
            ie, te = F.normalize(out.image_embeds, p=2, dim=1), \
                     F.normalize(out.text_embeds,  p=2, dim=1)
            sim   = (ie @ te.T) * clipM.logit_scale.exp()
            tgt   = torch.arange(sim.size(0), device=device)
            loss  = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) / 2
            loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0: print(f"ep{ep:02d}  loss {tot/len(loader):.4f}")
    clipM.eval(); clipM.save_pretrained("./clip_ft"); proc.save_pretrained("./clip_ft")

# ---------- 3.  retrieval helper ----------
def clip_guess(hints, choices, alpha=0.5):
    # encode hints → imgs + desc
    imgs = [ICON_DB[h]['icons'] for h in hints]
    desc = [f"an icon showing {ICON_DB[h]['description'].replace('\n',' and ')}"
            for h in hints]
    enc_h = proc(images=imgs, text=desc, return_tensors='pt',
                 padding=True, truncation=True).to(device)
    enc_c = proc(text=[f"a {c}" for c in choices], return_tensors='pt',
                 padding=True, truncation=True).to(device)

    with torch.no_grad():
        ih = F.normalize(clipM.get_image_features(**enc_h), p=2, dim=1)
        th = F.normalize(clipM.get_text_features(**enc_h),  p=2, dim=1)
        tc = F.normalize(clipM.get_text_features(**enc_c),  p=2, dim=1)
        sim = alpha * (ih @ tc.T) + (1-alpha) * (th @ tc.T)
        score = sim.sum(0)
        top = score.topk(10).indices.cpu()
    return [choices[i] for i in top]

</code></pre>

</details>



