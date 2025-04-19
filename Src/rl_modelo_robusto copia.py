# Importar librerías
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Carga de datos
ruta_absoluta = '/Users/kaabil/Documents/EDA Amazon/Data/Processed/amazon_dataset_final.csv'
df = pd.read_csv(ruta_absoluta)
X = df[['actual_price', 'rating']]
y = df['discounted_price']

# División de datos
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

# Ajuste del modelo OLS con errores estándar robustos (HC3)
X_train_const = sm.add_constant(X_train_s)
ols_model = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')

# Guardar scaler y modelo de forma robusta
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scaler_path = os.path.join(base_dir, "Data", "Preprocess", "scaler.pkl")
model_path = os.path.join(base_dir, "Models", "model.pkl")
joblib.dump(scaler, scaler_path)
joblib.dump(ols_model, model_path)