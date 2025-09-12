# Entrenamiento automático de modelos por categoría principal
import joblib, pathlib, numpy as np, pandas as pd
import re
import io
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# --- Configuración ---
MIN_REGISTROS = 500
DATA_DIR = "Data/Raw/Data Raw csv_files/"
MODEL_DIR = "Model"
target = "discount_price"
num_feats = ["actual_price", "ratings", "no_of_ratings"]
cat_feat = ["sub_category"]

# --- Funciones de Limpieza y Preprocesamiento ---

def clean_text(name):
    """Limpia texto para nombres de categoría."""
    name = str(name).strip().lower()
    name = re.sub(r'[^a-zA-Z0-9& ]', '', name)
    return name

def clean_numeric(value):
    """Limpia y convierte valores a numérico, manejando errores."""
    if pd.isnull(value):
        return np.nan
    value = str(value)
    # Elimina cualquier carácter que no sea dígito o punto decimal
    value = re.sub(r'[^\d.]', '', value)
    # Maneja el caso de múltiples puntos decimales, quedándose con el primero
    if value.count('.') > 1:
        parts = value.split('.')
        value = parts[0] + '.' + ''.join(parts[1:])
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def build_pipeline():
    """Construye el pipeline de preprocesamiento y el modelo."""
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

# --- Carga y Limpieza de Datos ---

print("Cargando y combinando archivos de datos...")
file_parts = [f"{DATA_DIR}Amazon-Products.csv.partaa", f"{DATA_DIR}Amazon-Products.csv.partab"]
full_content = ""
for part_path in file_parts:
    try:
        with open(part_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_content += f.read()
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo '{part_path}'.")

if not full_content:
    raise FileNotFoundError("No se encontraron archivos de datos para procesar. Verifica la ruta y existencia de los archivos .part*")

# Leer el contenido combinado en pandas
data_io = io.StringIO(full_content)

# Obtener cabeceras de un archivo de referencia
try:
    header_df = pd.read_csv(f"{DATA_DIR}All Appliances.csv", nrows=1)
    columns = header_df.columns
except Exception as e:
    print(f"Error al leer las cabeceras: {e}. Usando un conjunto de columnas predefinido.")
    columns = ['name', 'main_category', 'sub_category', 'image', 'link', 'ratings', 'no_of_ratings', 'discount_price', 'actual_price']

df = pd.read_csv(data_io, header=None, names=columns, sep=',', on_bad_lines='warn')

print("Limpiando datos...")
# Limpiar todas las columnas numéricas y el objetivo
for col in num_feats + [target]:
    df[col] = df[col].apply(clean_numeric)

# Eliminar filas con valores nulos en columnas clave
df.dropna(subset=num_feats + [target], inplace=True)

# Limpiar nombres de categorías
df["main_category"] = df["main_category"].apply(clean_text)
df["sub_category"] = df["sub_category"].apply(clean_text)

# --- Entrenamiento de Modelos ---

# Filtrar categorías relevantes
cat_counts = df["main_category"].value_counts()
relevant_cats = cat_counts[cat_counts >= MIN_REGISTROS].index.tolist()
print(f"Categorías seleccionadas (>{MIN_REGISTROS} registros):", relevant_cats)

pathlib.Path(MODEL_DIR).mkdir(exist_ok=True)
r2_results = {}

print("\nEntrenando modelos por categoría...")
for cat in relevant_cats:
    df_cat = df[df["main_category"] == cat].copy()
    
    X = df_cat[num_feats + cat_feat]
    y = df_cat[target]
    
    if y.empty or X.empty:
        print(f"Categoría '{cat}' no tiene datos suficientes después de la limpieza. Saltando.")
        continue

    pipe = build_pipeline()
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        r2 = scores.mean()
        r2_results[cat] = r2
        print(f"{cat:25s}  R² CV = {r2:.4f}")
        
        pipe.fit(X, y)
        fname = f"{MODEL_DIR}/model_{cat.replace(' ','_')}.pkl"
        joblib.dump(pipe, fname)
    except Exception as e:
        print(f"Error al entrenar el modelo para la categoría '{cat}': {e}")

# --- Modelo Global ---
print("\nEntrenando modelo global...")
X_global = df[num_feats + cat_feat]
y_global = df[target]

if not X_global.empty and not y_global.empty:
    pipe_global = build_pipeline()
    try:
        pipe_global.fit(X_global, y_global)
        joblib.dump(pipe_global, f"{MODEL_DIR}/model_global.pkl")
        print("Modelo global entrenado y guardado.")
    except Exception as e:
        print(f"Error al entrenar el modelo global: {e}")
else:
    print("No hay datos suficientes para entrenar el modelo global.")


print("\n--- Proceso Finalizado ---")
print("Modelos finales guardados en /Model/:")
for p in pathlib.Path(MODEL_DIR).glob("*.pkl"):
    print(" •", p.name)

print("\nResumen R² especialistas:", r2_results)
