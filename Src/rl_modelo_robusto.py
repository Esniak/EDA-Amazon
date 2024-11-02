
# Importar las librerías necesarias
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Paso 1: Importar el dataset
amazon_correl = pd.read_csv(r'/content/amazon_dataset_final.csv')

# Paso 2: Selección de características y variable objetivo (solo 'actual_price')
X = amazon_correl[['actual_price']]
Y = amazon_correl['discounted_price']

# Paso 3: Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Agregar una constante (intercepto) al modelo
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Paso 4: Calcular el VIF para cada variable predictora en el conjunto de entrenamiento
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train_const.values, i)
    for i in range(X_train_const.shape[1])
]

# Paso 5: Ajustar el modelo de regresión lineal (OLS) en el conjunto de entrenamiento
model_ols = sm.OLS(Y_train, X_train_const).fit()

# Ajustar el modelo con errores estándar robustos (HC3)
model_robust = model_ols.get_robustcov_results(cov_type='HC3')

# Paso 6: Obtener los coeficientes del modelo de forma segura
params_series = pd.Series(model_robust.params, index=model_robust.model.exog_names)

# Paso 7: Evaluar el modelo en el conjunto de prueba
y_pred_test = model_robust.predict(X_test_const)
y_true_test = Y_test

r_squared_test = r2_score(y_true_test, y_pred_test)
mse_test = mean_squared_error(y_true_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_true_test, y_pred_test)
durbin_watson_stat_test = sm.stats.durbin_watson(model_robust.resid)

# Paso 8: Test de Breusch-Pagan en el conjunto de entrenamiento
bp_test = het_breuschpagan(model_robust.resid, model_robust.model.exog)
bp_test_results = {
    'Lagrange multiplier statistic': bp_test[0],
    'p-value': bp_test[1],
    'f-value': bp_test[2],
    'f p-value': bp_test[3]
}

# Test de White en el conjunto de entrenamiento
white_test = het_white(model_robust.resid, model_robust.model.exog)
white_test_results = {
    'Test Statistic': white_test[0],
    'Test Statistic p-value': white_test[1],
    'F-Statistic': white_test[2],
    'F-Statistic p-value': white_test[3]
}

# Paso 9: Obtener los coeficientes del modelo de forma segura
required_params = ['const', 'actual_price']
missing_params = [param for param in required_params if param not in params_series.index]

if not missing_params:
    intercept = params_series['const']
    slope = params_series['actual_price']

    # Formatear la ecuación del modelo
    equation = f"I = {slope:.3f}X + {intercept:.3f}"
    interpretation = f"Esto indica que por cada incremento de 1 Euro en el precio real, el precio con descuento aumenta en {slope:.3f} Euros."

    # Almacenar la ecuación y la interpretación para uso futuro
    model_equation = equation
    model_interpretation = interpretation
else:
    available_params = params_series.index.tolist()
    raise KeyError(f"Parámetros faltantes: {missing_params}. Parámetros disponibles: {available_params}")

# Paso 10: Guardar el modelo en disco utilizando pickle
filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model_robust, file)