# Entrenamiento automático de modelos por categoría principal
import joblib, pathlib, numpy as np, pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# Configuración
MIN_REGISTROS = 500
DATA_PATH = "../Data/Raw/Data Raw csv_files/Amazon-Products.csv"
MODEL_DIR = "../Model"
target = "discount_price"
num_feats = ["actual_price", "rating", "no_of_ratings"]
cat_feat = ["sub_category"]

# Función para limpiar nombres
import re
def clean_cat(name):
    name = str(name).strip().lower()
    name = re.sub(r'[^a-zA-Z0-9& ]', '', name)
    return name

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

# Cargar datos
print("Cargando datos...")
df = pd.read_csv(DATA_PATH)
df["main_category"] = df["main_category"].apply(clean_cat)

# Filtrar categorías relevantes
cat_counts = df["main_category"].value_counts()
relevant_cats = cat_counts[cat_counts >= MIN_REGISTROS].index.tolist()
print(f"Categorías seleccionadas (>{MIN_REGISTROS} registros):", relevant_cats)

pathlib.Path(MODEL_DIR).mkdir(exist_ok=True)
r2_results = {}

for cat in relevant_cats:
    df_cat = df[df["main_category"] == cat].copy()
    # Asegurar columnas numéricas
    for col in num_feats:
        df_cat[col] = pd.to_numeric(df_cat[col], errors='coerce')
    # Crear columnas derivadas
    if "log_rating_count" not in df_cat.columns:
        df_cat["log_rating_count"] = np.log1p(df_cat["no_of_ratings"])
    if "price_per_rating" not in df_cat.columns:
        df_cat["price_per_rating"] = df_cat["actual_price"] / (df_cat["rating"] + 0.1)
    if "text_len_about" not in df_cat.columns:
        df_cat["text_len_about"] = 0
    X = df_cat[num_feats + cat_feat]
    y = df_cat[target]
    pipe = build_pipeline()
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    r2 = scores.mean()
    r2_results[cat] = r2
    print(f"{cat:25s}  R² CV = {r2:.4f}")
    pipe.fit(X, y)
    fname = f"{MODEL_DIR}/model_{cat.replace(' ','_')}.pkl"
    joblib.dump(pipe, fname)

# Modelo global
X_global = df[num_feats + cat_feat]
y_global = df[target]
pipe_global = build_pipeline()
pipe_global.fit(X_global, y_global)
joblib.dump(pipe_global, f"{MODEL_DIR}/model_global.pkl")

print("\nModelos finales guardados en /Model/:")
for p in pathlib.Path(MODEL_DIR).glob("*.pkl"):
    print(" •", p.name)

print("\nResumen R² especialistas:", r2_results)
