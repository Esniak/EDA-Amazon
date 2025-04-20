# app5modelos.py — Dashboard Streamlit (4 especialistas + global, versión pro)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, glob, os, time
import matplotlib.pyplot as plt

# ---------- Página ----------
st.set_page_config(page_title="Precio Promocional Amazon ES",
                   page_icon="💸", layout="wide")

# ---------- Cargar modelos ----------
def cargar_modelos(dir_path="Model"):
    especialistas, global_model = {}, None
    for pkl in glob.glob(os.path.join(dir_path, "model_*.pkl")):
        base = os.path.basename(pkl)
        if base == "model_global.pkl":
            global_model = joblib.load(pkl)
        else:
            key = base.replace("model_", "").replace(".pkl", "").replace("_", "&")
            especialistas[key] = joblib.load(pkl)
    return especialistas, global_model

modelos_especialistas, modelo_global = cargar_modelos()

# Depuración: mostrar modelos cargados
st.write("Modelos especialistas disponibles:", list(modelos_especialistas.keys()))
st.write("¿Modelo global cargado?", modelo_global is not None)
if not modelos_especialistas and not modelo_global:
    st.error("No se han cargado modelos. Revisa la carpeta Models/ y los archivos .pkl.")

# ---------- Estilos ----------
st.markdown("""
<style>
body  {background-color:#FAFAFA;}
div[data-testid="stMetric"] > label                 {font-weight:600;}
div[data-testid="stMetric"] > div > div:nth-child(1){color:#0D47A1;}
div[data-testid="stMetric"] > div > div:last-child  {color:#004D40;}
hr   {margin:0.3rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("🛠️ Parámetros")
main_cat = st.sidebar.selectbox(
    "Categoría principal",
    list(modelos_especialistas.keys())
)
sub_cat = st.sidebar.text_input("Sub‑categoría", "General")
precio_base = st.sidebar.number_input("Precio base (€)", 0.0, 9999.0, 59.99, 0.10)
rating = st.sidebar.slider("Rating (★)", 1.0, 5.0, 4.2, 0.1)
rating_count = st.sidebar.number_input("Nº reseñas", 0, 500000, 1500, 10)
descripcion = st.sidebar.text_area("Descripción breve (opcional)")
text_len_about = len(descripcion)

if "pred" not in st.session_state:
    st.session_state.pred = None

# ---------- Tabs ----------
tabs = st.tabs(["📈 Predicción", "ℹ️ About"])

with tabs[0]:
    st.title("💸 Recomendador de Precio en Oferta")

    if st.button("Calcular precio con descuento"):
        with st.spinner("Calculando…"):
            price_per_rating = precio_base / (rating + 0.1)
            log_rating_count = np.log1p(rating_count)

            df_in = pd.DataFrame([{
                "actual_price_eur":     precio_base,
                "rating":               rating,
                "log_rating_count":     log_rating_count,
                "price_per_rating":     price_per_rating,
                "text_len_about":       text_len_about,
                "sub_category":         sub_cat
            }])

            modelo = modelos_especialistas.get(main_cat, None)
            aviso = f"Modelo especialista: **{main_cat}**"
            precision_txt = "Precisión estimada ± 10 %"
            if modelo is None:
                if modelo_global is not None:
                    modelo = modelo_global
                    aviso = "⚠️ Modelo genérico (datos limitados)"
                    precision_txt = "Precisión estimada ± 25 %"
                else:
                    st.error("No se encontró un modelo para la categoría seleccionada ni modelo global. No se puede predecir.")
                    st.session_state.pred = None
                    st.stop()

            # Simulación de loading
            time.sleep(0.2)
            try:
                precio_desc = float(modelo.predict(df_in)[0])
            except Exception as e:
                st.error(f"Error al predecir: {e}")
                st.session_state.pred = None
                st.stop()
            st.session_state.pred = (precio_desc, aviso, precision_txt)

    # ------------ Mostrar resultado ------------
    if st.session_state.pred:
        precio_desc, aviso, precision_txt = st.session_state.pred
        descuento_pct = 100 * (1 - precio_desc / precio_base) if precio_base else 0

        st.markdown(aviso)
        st.markdown(precision_txt)
        colA, colB = st.columns(2)
        colA.metric("Precio recomendado (€)", f"{precio_desc:,.2f}")
        colB.metric("Descuento (%)", f"{descuento_pct:,.1f}")

        st.divider()
        fig, ax = plt.subplots(figsize=(5, 1.8))
        ax.barh(["Actual", "Recomendado"], [precio_base, precio_desc], color=["#1E88E5", "#43A047"])
        ax.set_xlabel("€"); ax.set_xlim(0, max(precio_base, precio_desc) * 1.2)
        for spine in ["top","right","left"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, clear_figure=True)

with tabs[1]:
    st.subheader("Cómo funciona")
    st.markdown("""
    1. Selecciona un modelo **especialista** si tu categoría lo tiene.<br>
    2. De lo contrario, aplica un **modelo global** entrenado con todo el catálogo.<br>
    3. El pipeline incluye escalado, _target encoding_ y predicción XGBoost.<br>
    4. El margen de error mostrado proviene de la validación cruzada.<br><br>
    **Variables usadas**  
    • Precio base en € • Rating y nº reseñas • Longitud descripción • Sub‑categoría
    """, unsafe_allow_html=True)