# KS Intelligence – Amazon Price Optimizer Dashboard
# ---------------------------------------------------
# Este script crea una aplicación Streamlit que permite introducir el precio actual
# y la valoración (rating) de un producto y devuelve el precio descontado óptimo
# utilizando el modelo OLS entrenado y el scaler guardado.
# Ambos archivos (model.pkl y scaler.pkl) deben estar en las rutas especificadas.

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import statsmodels.api as sm
import shutil

# --------------------------------------------------
# Unir archivos divididos automáticamente
# --------------------------------------------------
PART1 = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Raw', 'Data Raw csv_files', 'Amazon-Products.csv.partaa')
PART2 = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Raw', 'Data Raw csv_files', 'Amazon-Products.csv.partab')
FULL = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Raw', 'Data Raw csv_files', 'Amazon-Products.csv')

if os.path.exists(PART1) and os.path.exists(PART2):
    # Solo unir si el archivo completo no existe
    if not os.path.exists(FULL):
        with open(FULL, 'wb') as f_out:
            for part in [PART1, PART2]:
                with open(part, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
        print(f"Archivo unido correctamente: {FULL}")

# --------------------------------------------------
# Utilidades
# --------------------------------------------------
@st.cache_resource  # carga una única vez
def load_artifacts(model_path: str = None, scaler_path: str = None):
    """Carga modelo y scaler desde rutas absolutas relativas a este archivo, robusto a cualquier working directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = model_path or os.path.join(base_dir, "Models", "model.pkl")
    scaler_path = scaler_path or os.path.join(base_dir, "Data", "Preprocess", "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_dataset(csv_path: str | None = None):
    """Carga el dataset de entrenamiento si está disponible para mostrar métricas / gráficos."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return None

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Amazon Price Optimizer – KS Intelligence",
    layout="wide",
    page_icon="🛒",
)

st.title("🛒 Amazon Price Optimizer – KS Intelligence")
st.markdown(
    """
    Introduce **Precio actual (€)** y **Rating (1–5)** del producto para obtener la recomendación de
    **precio con descuento óptimo** basada en nuestro modelo de regresión (OLS + StandardScaler).
    """
)

# Cargar artefactos
MODEL_PATH = None  # Usar None para que la función calcule la ruta robusta
SCALER_PATH = None
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# Sidebar – hiperparámetros del input
with st.sidebar:
    st.header("Parametros de entrada")
    actual_price = st.number_input("Precio Actual (€)", min_value=0.0, step=0.1, value=100.0)
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1, value=4.5)
    submit = st.button("Calcular precio con descuento", type="primary")

# Predicción
if submit:
    X_new = np.array([[actual_price, rating]])
    X_scaled = scaler.transform(X_new)
    X_scaled_const = sm.add_constant(X_scaled, has_constant='add')
    discounted_price_pred = float(model.predict(X_scaled_const)[0])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio actual (€)", f"{actual_price:,.2f}")
    with col2:
        st.metric("Precio con descuento sugerido (€)", f"{discounted_price_pred:,.2f}")

    st.success("Recomendación generada con éxito.")

# --------------------------------------------------
# Análisis y visualización opcional
# --------------------------------------------------
DATA_PATH = os.path.join("data", "amazon_dataset_final.csv")  # ajusta si es necesario

df_train = load_dataset(DATA_PATH)

if df_train is not None:
    import altair as alt

    st.subheader("Distribución histórico de precios vs. precios descontados")
    chart = (
        alt.Chart(df_train.sample(min(2000, len(df_train))), height=400)
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X("actual_price", title="Precio actual (€)"),
            y=alt.Y("discounted_price", title="Precio con descuento (€)"),
            color=alt.value("#4B9CD3"),
            tooltip=["actual_price", "rating", "discounted_price"]
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Resumen estadístico del conjunto de entrenamiento")
    st.dataframe(df_train.describe().T, use_container_width=True)
else:
    st.info("Carga el dataset en la carpeta 'data' para ver análisis adicionales.")

st.caption("© 2025 KS Intelligence – Optimización de precios con IA")
