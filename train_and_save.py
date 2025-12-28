# train_and_save.py
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Paths
ROOT = Path.cwd()
DATA_DIR = ROOT / "data_processed"
MODEL_DIR = ROOT / "backend" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = DATA_DIR / "train.parquet"
FEATURE_LIST = DATA_DIR / "feature_list.json"
IMPUTER_PATH = DATA_DIR / "imputer.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"

# Settings
SAMPLE_FRAC = 0.05        # Use a 5% sample for quick training
CAT_ITER = 500            # Increase for better accuracy (e.g., 1500–3000)
RANDOM_STATE = 42
TARGET_COL = "RET"

# ----------------------- Load Feature List -----------------------
with open(FEATURE_LIST, "r") as f:
    features = json.load(f)

print(f"Loaded {len(features)} features.")

# ----------------------- Load Training Data -----------------------
print("Loading training parquet (this may take a moment)...")
df = pd.read_parquet(TRAIN_PARQUET)

if SAMPLE_FRAC < 1.0:
    df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Using sampled data: {len(df)} rows.")
else:
    print(f"Using full dataset: {len(df)} rows.")

X = df[features].copy()
y = df[TARGET_COL].copy()

# ----------------------- Imputer -----------------------
if IMPUTER_PATH.exists():
    print("Loading imputer.pkl")
    imputer = joblib.load(IMPUTER_PATH)
else:
    print("Creating new median SimpleImputer.")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X)

X_imp = pd.DataFrame(imputer.transform(X), columns=features)

# ----------------------- Scaler-----------------------
if SCALER_PATH.exists():
    print("Loading scaler.pkl")
    scaler = joblib.load(SCALER_PATH)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=features)
else:
    print("No scaler found — using imputed values directly.")
    scaler = None
    X_scaled = X_imp

# Save imputer/scaler into backend/model folder
joblib.dump(imputer, MODEL_DIR / "imputer.pkl")
if scaler is not None:
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# ----------------------- Train/Val Split -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

# ----------------------- Train CatBoost -----------------------
print("Training CatBoost...")

cat_params = {
    "iterations": CAT_ITER,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": RANDOM_STATE,
    "verbose": 50,
    "task_type": "CPU",
}

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

cat = CatBoostRegressor(**cat_params)
cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

# ----------------------- Save Model -----------------------
cbm_path = MODEL_DIR / "catboost_model.cbm"
pkl_path = MODEL_DIR / "catboost_model.pkl"

cat.save_model(str(cbm_path))
joblib.dump(cat, pkl_path)

print(f"\nSaved CatBoost model to:\n - {cbm_path}\n - {pkl_path}")
print("\nTraining complete.")
