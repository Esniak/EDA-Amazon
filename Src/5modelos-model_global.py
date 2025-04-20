# ===========================================================
# ENTRENAR y GUARDAR 4 especialistas + modelo global
# ===========================================================
import joblib, pathlib, numpy as np, pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# ---------- Configuración ----------
target = "discounted_price_eur"
num_feats = [
    "actual_price_eur", "rating", "log_rating_count",
    "price_per_rating", "text_len_about"
]
cat_feat = ["sub_category"]

def build_pipeline():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", TargetEncoder(cols=cat_feat), cat_feat)
    ])
    model = XGBRegressor(
        n_estimators=600, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    return Pipeline([("prep", preprocess), ("model", model)])

# ---------- Categorías escogidas ----------
keep_cats = [
    "Home&Kitchen",
    "Computers&Accessories",
    "OfficeProducts",
    "Electronics"
]

# ---------- Cargar datos ----------
df = pd.read_csv("Data/Processed/amazon_clean.csv")
if "main_category" not in df.columns:
    raise ValueError("La columna 'main_category' no existe en el DataFrame. Revisa el archivo de datos.")

# Asegurar que las columnas numéricas son float
for col in ["actual_price_eur", "rating", "rating_count"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Crear columnas derivadas si no existen
def safe_log1p(x):
    try:
        return np.log1p(x)
    except Exception:
        return 0

if "log_rating_count" not in df.columns:
    df["log_rating_count"] = df["rating_count"].apply(safe_log1p) if "rating_count" in df.columns else 0
if "price_per_rating" not in df.columns:
    df["price_per_rating"] = df["actual_price_eur"] / (df["rating"] + 0.1)
if "text_len_about" not in df.columns:
    if "about_product" in df.columns:
        df["text_len_about"] = df["about_product"].fillna("").apply(len)
    else:
        df["text_len_about"] = 0

pathlib.Path("Model").mkdir(exist_ok=True)
r2_results = {}

for cat in keep_cats:
    df_cat = df[df["main_category"] == cat].copy()
    X = df_cat[num_feats + cat_feat]
    y = df_cat[target]

    pipe = build_pipeline()
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    r2 = scores.mean()
    r2_results[cat] = r2
    print(f"{cat:25s}  R² CV = {r2:.4f}")

    pipe.fit(X, y)
    fname = f"Model/model_{cat.replace('&','_')}.pkl"
    joblib.dump(pipe, fname)

# ---------- Modelo global ----------
X_global = df[num_feats + cat_feat]
y_global = df[target]
pipe_global = build_pipeline()
pipe_global.fit(X_global, y_global)
joblib.dump(pipe_global, "Model/model_global.pkl")

print("\nModelos finales guardados en /Model/:")
for p in pathlib.Path("Model").glob("*.pkl"):
    print(" •", p.name)

print("\nResumen R² especialistas:", r2_results)
