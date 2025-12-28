from catboost import CatBoostClassifier
from pathlib import Path
import json

MODEL_DIR = Path("backend/model")

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "catboost_model.cbm"))

# --- CORRECT way ---
model_features = model.feature_names_

with open(MODEL_DIR / "feature_list.json") as f:
    json_features = json.load(f)

print("Model feature count:", len(model_features))
print("feature_list.json count:", len(json_features))

print("\nIn model but NOT in feature_list.json:")
print(set(model_features) - set(json_features))

print("\nIn feature_list.json but NOT in model:")
print(set(json_features) - set(model_features))
