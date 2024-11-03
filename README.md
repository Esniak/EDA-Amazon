# Análisis de Datos Exploratorios de Ventas de Amazon (EDA)

![Descripción de la imagen](https://drive.google.com/uc?export=view&id=1GLplIyn0_KiOJ6IMd9FOFKWx6mJmJxCR)

# Resumen del Proyecto

Este proyecto se centra en realizar un Análisis Exploratorio de Datos (EDA) sobre las ventas de productos en Amazon. El objetivo principal es identificar métricas clave como las categorías de productos con mayores ingresos, los descuentos más significativos y los niveles de compromiso del cliente, entre otros aspectos relevantes. A través de este análisis, se busca proporcionar recomendaciones y conclusiones que faciliten la toma de decisiones empresariales, orientando estrategias de ventas y marketing.

La solución obtenida permitirá identificar patrones y tendencias en las ventas, ofreciendo una base para optimizar estrategias de marketing, mejorar la gestión del inventario, y aumentar tanto las ventas como la satisfacción del cliente. Las visualizaciones y modelos de regresión desarrollados proporcionarán información valiosa para los tomadores de decisiones, contribuyendo así al crecimiento y éxito de la empresa en el mercado.

## Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [Resumen del código](#resumen-del-codigo)
4. [Methodology](#methodology)
5. [Analysis](#analysis)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Recommendations](#recommendations)
9. [Deployment](#deployment)
10. [Link notebook completo en inglés](#link-notebook-completo-en-ingles)

## Introducción

En este proyecto, llevamos a cabo un análisis detallado de las ventas de productos en Amazon para descubrir información crucial que ayude a comprender el rendimiento de los productos, las preferencias de los clientes y la efectividad de las estrategias de precios y descuentos. A través de la exploración de datos históricos, buscamos identificar tendencias que no solo expliquen el comportamiento del cliente, sino que también ofrezcan herramientas prácticas para optimizar las decisiones empresariales.

El análisis se enfocará en áreas como el impacto de los descuentos en las ventas, la correlación entre el nivel de compromiso del cliente y las métricas de rendimiento de los productos, y la identificación de patrones relevantes para mejorar la estrategia de ventas. Además, se han desarrollado modelos de regresión para analizar las relaciones entre diferentes variables y proporcionar predicciones útiles que apoyen la toma de decisiones. Esta introducción plantea las bases para el análisis y define el marco que utilizarán los tomadores de decisiones para implementar estrategias de crecimiento.

## Datos

El conjunto de datos utilizado en este proyecto incluye información detallada sobre más de 1000 productos vendidos en Amazon, abarcando aspectos como el precio, la calificación, las reseñas de los clientes, y la descripción de los productos. Los datos para este análisis están almacenados en el archivo `amazon.csv`.link: https://github.com/Esniak/-EDA-Amazon-Sales/tree/main/Data/Raw

### Archivo de Datos

- **Nombre del Archivo**: amazon.csv
- **Número de Columnas**: 16

### Descripción de las Columnas

- **Product_id**: Identificación única del producto.
- **Product_name**: Nombre del producto.
- **Category**: Categoría a la que pertenece el producto.
- **Discounted_price**: Precio del producto después de aplicar el descuento.
- **Actual_price**: Precio original del producto sin descuento.
- **Discount_percent**: Porcentaje de descuento aplicado sobre el precio original.
- **Rating**: Calificación promedio del producto (escala de 0 a 5 estrellas).
- **Rating_count**: Número de clientes que han calificado el producto.
- **About_product**: Descripción del producto.
- **User_id**: Identificación única del usuario.
- **User_name**: Nombre del usuario.
- **Review_id**: Identificación única de la reseña.
- **Review_title**: Título de la reseña.
- **Review_content**: Contenido de la reseña.
- **Img_link**: Enlace a la imagen del producto.
- **Product_link**: Enlace a la página del producto en Amazon.

## Resumen del Código

Este proyecto está compuesto por cuatro notebooks principales que abarcan limpieza y transformación de datos, análisis exploratorio y visualización, modelado de regresión lineal, y análisis adicional de regresión Lineal Robusto. A continuación, se presenta un resumen conciso de las funcionalidades y estructuras de código empleadas en cada notebook.

**Notebook 1: Limpieza y Transformación de Datos**

- **Importación de Librerías y Carga de Datos**: Se importó únicamente la librería `pandas`. Los datos se cargaron desde `amazon.csv` utilizando `pd.read_csv()`.
- **Exploración y Limpieza de Datos**: Se utilizó `.head()`, `.describe()`, y `.info()` para la exploración inicial. Las columnas de precio (`discounted_price`, `actual_price`) se limpiaron con `.str.replace()`, se convirtieron a tipo `float64` con `.astype()`, y se transformaron a euros mediante funciones `lambda` aplicadas con `.apply()`.
- **Transformaciones Adicionales**: Se creó `discount_amount` calculando la diferencia entre `actual_price` y `discounted_price`, y `discount_percent` se limpió eliminando `%` y dividiendo por 100.
- **Manejo de Valores Nulos y Duplicados**: Se identificaron valores nulos con `.isna().sum()`, se reemplazaron en `rating_count` con `.fillna()`, y se eliminaron filas duplicadas con `.drop_duplicates()`.
- **Guardado de Datos**: Se guardaron los dataframes limpios como `amazon_clean.csv` y `reviewers_data.csv` usando `.to_csv()`.
  
- **link notebook 1:**  https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/exploratory_data_analysis.ipynb

**Notebook 2: Análisis Exploratorio y Visualización de Datos**

- **Visualizaciones Clave**: Se realizaron análisis de precios y descuentos con gráficos de barras (`seaborn.barplot()`), matriz de correlación con `.corr()` visualizada mediante `seaborn.heatmap()`, y gráficos de dispersión para analizar relaciones entre precios (`seaborn.scatterplot()`).
- **Distribución y Comparación de Variables**: Se generaron histogramas con `seaborn.histplot()` para la distribución de `rating`, `actual_price`, y `discounted_price`, y se analizaron porcentajes de productos por categoría con `.value_counts()` y `.groupby()`.
  
- **link notebook 2:**  https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/exploratory_data_visualizations.ipynb
  
**Notebook 3: Análisis de Regresión Lineal**

- **Entrenamiento del Modelo**: Se utilizó `LinearRegression()` de `sklearn` para modelar `actual_price` en función de variables como `discounted_price` y `rating`. Los datos se dividieron en entrenamiento y prueba con `train_test_split()`, y el rendimiento se evaluó mediante métricas como R² y RMSE.
- **Validación y Supuestos del Modelo**: Se aplicó validación cruzada con `cross_val_score()`, y se verificaron los supuestos del modelo utilizando `statsmodels` para detectar heteroscedasticidad y autocorrelación.
  
- **link notebook 3:** https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/RL_modelo.ipynb
  
**Notebook 4: Análisis Regresión Lineal modelo robusto**

- **Ajuste del Modelo y Evaluación**: Se ajustó un modelo OLS con `statsmodels`, y se evaluaron métricas como R² y MAE sobre el conjunto de prueba. Se utilizó `variance_inflation_factor()` para verificar la multicolinealidad.
- **Modelo Robusto**: Finalmente, se desarrolló un modelo de regresión robusto para corregir posibles problemas de heteroscedasticidad y mejorar la fiabilidad del análisis. Se obtuvieron resultados robustos utilizando errores estándar HC3 con `get_robustcov_results()`.
- **Visualización de Residuos**: Se generaron gráficos de residuos y se realizaron pruebas de heteroscedasticidad (Breusch-Pagan y White) para asegurar la validez del modelo.
  
- **link notebook 4:**  https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/RL_Modelo_Robusto.ipynb
  
## Metodología

La metodología de este proyecto incluye los siguientes pasos:

1. **Carga de Datos**: Importación del dataset `amazon.csv` utilizando `pandas` (`pd.read_csv()`), junto con la importación de librerías necesarias como `numpy`, `matplotlib.pyplot` y `seaborn`.

2. **Análisis Exploratorio Inicial**: Se utilizaron métodos como `.head()`, `.describe()`, y `.info()` para comprender la estructura de los datos y la distribución de las variables. Además, se analizaron correlaciones entre diferentes variables utilizando `.corr()` y se visualizaron con `seaborn.heatmap()`.

3. **Limpieza de Datos**: Incluyó la eliminación de caracteres innecesarios en columnas de precios mediante `.str.replace()`, conversión de tipos de datos (`.astype()`), y manejo de valores nulos (`.isna().sum()`, `.fillna()`). También se eliminaron duplicados con `.drop_duplicates()`, asegurando la integridad de los datos.

4. **Transformación de Datos**: Las columnas `discounted_price` y `actual_price` fueron transformadas para convertir los valores a euros aplicando una conversión con funciones lambda (`.apply()`). Se añadieron columnas calculadas como `discount_amount` y `discount_percent` para enriquecer el análisis.

5. **Visualización de Datos**: Creación de gráficos de barras (`seaborn.barplot()`), gráficos de dispersión (`seaborn.scatterplot()`), y histogramas (`seaborn.histplot()`) para analizar las tendencias y distribuciones de precios, descuentos y calificaciones de productos.

6. **Modelado y Análisis de Regresión**: Se desarrollaron modelos de regresión lineal utilizando `LinearRegression()` y `statsmodels` para comprender las relaciones entre las variables clave e identificar patrones que apoyen las decisiones de negocio. **Finalmente, se creó un modelo de regresión robusto** para abordar posibles problemas de heteroscedasticidad, asegurando así la fiabilidad de los resultados. Se evaluaron las métricas del modelo y se aseguraron los supuestos fundamentales mediante validación cruzada y pruebas de heteroscedasticidad.

## Análisis

El análisis se centra en las siguientes áreas clave:

**Matriz de Correlación**: Creación de una matriz de correlación (`.corr()`) visualizada con `seaborn.heatmap()` para analizar cómo se relacionan las diferentes variables del conjunto de datos.

![Matriz de Correlación](https://drive.google.com/uc?export=view&id=1CYL9AQFprTlbl2WU3NEpf39NIL0es3-u)

**Gráfico de Dispersión**: Se generó un gráfico de dispersión con `seaborn.scatterplot()` para visualizar la correlación entre el precio real (`actual_price`) y el precio con descuento (`discounted_price`). Esto permitió analizar cómo los descuentos influyen en el precio final.

![Grafico de dispersion](https://drive.google.com/uc?export=view&id=1AumSbwx5p1k7ziyQMIFOV22t3L67Kl8f)

**Ingresos por Categoría**: Se identificaron las categorías de productos con mayores ingresos mediante agrupaciones (`.groupby()`) y análisis comparativos. Esto permitió visualizar qué categorías generan más valor, enfocado a decisiones de marketing.

**Distribución del porcentaje de descuento por categoría y subcategoría.**
La distribución del porcentaje de descuento por categoría y subcategoría revelará qué categorías y subcategorías ofrecen el mayor rango de porcentajes de descuento. También puede revelar qué categorías y subcategorías ofrecen el mayor porcentaje promedio de descuentos.

![Descuentos por Subcategoria](https://drive.google.com/uc?export=view&id=1Q32qz5DbhUlC8RYvCGrRIdSKrSFoeW36)

**Interacción del Cliente**: Evaluación de la interacción del cliente a través de `rating_count` y `rating`. Se utilizaron histogramas y curvas de densidad (`seaborn.histplot()`) para entender cómo las calificaciones afectan el compromiso de los clientes.

## Resultados

El análisis realizado produjo varios hallazgos clave que proporcionan una visión profunda de las ventas en Amazon y del comportamiento del cliente. A continuación se presentan los principales resultados:

**Categorías con Mayores Ingresos**: Las categorías de productos relacionadas con electrónica y artículos del hogar mostraron los mayores ingresos, lo cual fue evidente a partir de la agrupación y comparación de las ventas. Esto indica que las estrategias de marketing deben priorizar estos sectores para maximizar los beneficios.

**Efecto de los Descuentos**: Se observó que las categorías con descuentos más agresivos, como accesorios y moda, generaron un volumen de ventas mayor. Sin embargo, la rentabilidad de estos descuentos varía, ya que en algunos casos no se traduce directamente en mayores ingresos. Los gráficos de barras (`seaborn.barplot()`) ayudaron a ilustrar esta tendencia.

**Compromiso del Cliente**: Los productos con mayores `rating_count` y `rating` mostraron una correlación positiva con las ventas. Los histogramas y gráficos de densidad revelaron que la mayoría de los productos con mayores calificaciones también presentaron un alto volumen de ventas, indicando un mayor compromiso del cliente.

![calificación promedio por categoría y subcategoría](https://drive.google.com/uc?export=view&id=1XQuMcgWaDOHVJoChzSGZkSXxpp4E5mPa)

4. **Predicciones de Precios con el Modelo de Regresión**: El modelo de regresión lineal mostró que `actual_price` y `discount_percentage`, son factores significativos para predecir los precios con descuento. Estos factores demostraron ser consistentes en el análisis de validación cruzada, lo que proporciona una fuerte confianza en la fiabilidad del modelo.

## Conclusión

Esta conclusión analiza diversas categorías de productos vendidos en Amazon, enfocándose en precios, descuentos, compromiso del cliente y reseñas.

**Principales Hallazgos**

**1. Precios y Descuentos**
- **Electrónica**: Tiene el precio medio más alto por producto antes y después del descuento.
- **Hogar y cocina** y **Coche y moto**: Son las siguientes con precios medios elevados. Antes de los descuentos, ambas categorías tienen precios similares, pero después de aplicar los descuentos, "Hogar y cocina" ofrece descuentos más altos que "Coche y moto".
- **Descuentos por Categoría**:
  - **Computers & Accessories**: Ofrece el mayor descuento promedio del 54.02%.
  - **Electronics**: Descuento promedio del 50.83%.
  - La mayoría de los descuentos están en el rango del 40% al 70%.
  - Subcategorías con mayores porcentajes de descuento: Tecnología portátil, Auriculares y accesorios.

**2. Compromiso del Cliente**
- **Participación del Cliente**:
  - Las categorías con mayor participación son **Electronics**, **Home & Kitchen** y **Computers & Accessories**, representando el 97% de la variedad de productos y con el mayor número de reseñas.
- **Satisfacción del Cliente**:
  - Las categorías con mayor satisfacción son **Office Products** (calificación promedio de 4.31) y **Juguetes y juegos**.
  - La subcategoría **Tabletas** tiene la mayor satisfacción del cliente.
- La mayoría de los productos tienen una calificación de 4 a 4.5 estrellas y los clientes generalmente permanecen anónimos al dejar reseñas.

**Recomendaciones**

**Estrategias de Marketing y Promoción**:
- Continuar invirtiendo en Electrónica y Hogar&Cocina para mantener e incrementar los ingresos.
- Evaluar las políticas de descuento en Computadoras&Accesorios y Electrónica para optimizar la rentabilidad.

**Optimización de Descuentos**:
- Analizar si los altos descuentos en Computadoras&Accesorios y Electrónica atraen suficiente volumen de ventas.
- Experimentar con diferentes estrategias de descuento para encontrar un equilibrio óptimo entre volumen de ventas y márgenes de beneficio.

**Interacción del Cliente**:
- Mantener la calidad de productos y servicios en Oficina y Juguetes & Juegos.
- Mejorar la calidad del producto en Automóvil&Moto y otras categorías con menor calificación.

**Análisis de Reseñas**:
- Realizar un análisis de texto de las reseñas para identificar áreas específicas de mejora e innovación.
- Implementar un sistema de retroalimentación para que los clientes expresen sus opiniones de manera efectiva.

# Conclusiones sobre el Modelo

![Metrica RL modelo robusto](https://drive.google.com/uc?export=view&id=1erQGjtuLkpz7_Qz93ne0rDzNqiGm6yfM)

Basándonos en los resultados del modelo de regresión lineal Robusto, muestra un desempeño estadístico sólido. Los altos valores de R² en ambos conjuntos de datos (0.927 en entrenamiento y 0.9186 en prueba) indican que el modelo explica aproximadamente el 92% de la variabilidad en el precio con descuento basándose en el precio real. Esto sugiere que el modelo tiene una buena capacidad predictiva.

El coeficiente de `actual_price` es 0.6126, lo que significa que por cada incremento de 1 € en el precio real, el precio con descuento aumenta en aproximadamente 0.613 €. Este incremento menos que proporcional puede ser interesante desde una perspectiva empresarial, ya que indica que los descuentos no son lineales en relación con el precio original.

**Aplicabilidad Empresarial**

- **Estrategias de Precios**: Si una empresa está buscando establecer políticas de descuento basadas en el precio original, este modelo puede ayudar a predecir cómo ajustar los descuentos en función de los cambios en los precios.
- **Segmentación de Mercado**: Comprender esta relación puede ser útil para segmentar productos o clientes según su sensibilidad al precio y optimizar las ofertas promocionales.
- **Análisis de Competitividad**: El modelo puede servir para analizar cómo los precios con descuento se comparan con los de la competencia, permitiendo ajustar estrategias para mejorar la posición en el mercado.

**Consideraciones Adicionales**

- **Variables Omitidas**: Si bien el modelo muestra un buen ajuste, podría beneficiarse de la inclusión de otras variables que afecten el precio con descuento, como promociones especiales, estacionalidad, o características del producto.
- **Validación Continua**: Es importante mantener una validación continua del modelo para asegurarse de que sigue siendo relevante en condiciones cambiantes del mercado.

El resultado obtenido ofrece una visión valiosa que puede ser aplicada en la toma de decisiones empresariales. Dado su alto poder explicativo y la relevancia del coeficiente obtenido, el modelo puede ser una herramienta útil para optimizar estrategias de precios y descuentos.


## Deployment (Puesta en marcha)

El modelo de regresión lineal Robusto, se guardó en disco para su futuro uso mediante la librería `pickle`. Esto permite reutilizar el modelo sin necesidad de volver a entrenarlo, facilitando su integración en sistemas de recomendación o aplicaciones web. El siguiente código fue utilizado para guardar el modelo:

```python
# Guardar el modelo en disco utilizando pickle
filename = 'model.pkl'
try:
    with open(filename, 'wb') as file:
        pickle.dump(model_robust, file)
    print(f"\nModelo guardado en {filename}")
except Exception as e:
    print(f"\nError al guardar el modelo: {e}")
    raise
```

## Link Notebook completo en Español y Inglés
## Complete Notebook Link in Spanish and English

Si prefieres revisar el contenido en inglés o tienes alguna dificultad con la versión en español, puedes consultar el notebook completo en Inglés o en Español en los enlaces que te dejo a continuación:

[Notebook Completo en Español](https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/(EDA)_Ventas_Amazon_EN.ipynb)

If you prefer to review the content in English or encounter any difficulties with the Spanish version, you can consult the complete notebook in English or Spanish using the links provided below:

[Complete Notebook Link in English](https://github.com/Esniak/-EDA-Amazon-Sales/blob/main/Notebooks/(EDA)_Ventas_Amazon_EN.ipynb)



