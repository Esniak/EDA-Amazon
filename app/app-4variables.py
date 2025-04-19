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

# --------------------------------------------------
# Utilidades
# --------------------------------------------------
@st.cache_resource  # carga una única vez
def load_artifacts(model_path: str = None, scaler_path: str = None):
    """Carga modelo y scaler desde rutas absolutas relativas a este archivo, robusto a cualquier working directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = model_path or os.path.join(base_dir, "Models", "model-4variables.pkl")
    scaler_path = scaler_path or os.path.join(base_dir, "Data", "Preprocess", "scaler-4variables.pkl")
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
    Introduce los valores de **precio con descuento**, **% descuento**, **rating** y **número de valoraciones** del producto para obtener la predicción del **precio original** basada en nuestro modelo de regresión (OLS + StandardScaler).
    """
)

# Cargar artefactos
MODEL_PATH = None  # Usar None para que la función calcule la ruta robusta
SCALER_PATH = None
model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)

# Sidebar – hiperparámetros del input
with st.sidebar:
    st.header("Parámetros de entrada")
    discounted_price = st.number_input("Precio con descuento (€)", min_value=0.0, step=0.1, value=80.0)
    discount_percentage = st.number_input("% Descuento", min_value=0.0, max_value=100.0, step=0.1, value=20.0)
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1, value=4.5)
    rating_count = st.number_input("Número de valoraciones", min_value=0, step=1, value=100)
    submit = st.button("Calcular precio original", type="primary")

# Predicción
if submit:
    X_new = np.array([[discounted_price, discount_percentage, rating, rating_count]])
    X_scaled = scaler.transform(X_new)
    actual_price_pred = float(model.predict(X_scaled)[0])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio con descuento (€)", f"{discounted_price:,.2f}")
    with col2:
        st.metric("Precio original estimado (€)", f"{actual_price_pred:,.2f}")

    st.success("Predicción generada con éxito.")

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
