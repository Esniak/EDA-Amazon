
import pandas as pd

# utilizo el comando .read_csv para importar el archivo .csv.
amazon = pd.read_csv(r'/content/amazon.csv')

# para eliminar las comas y ₹ de las columnas discounted_price y actual_price, utilizo  el comando .str.replace()

amazon['discounted_price'] = amazon['discounted_price'].str.replace("₹",'').str.replace(",",'').astype('float64')

# para convertir el tipo de dato en las columnas discounted_price y actual_price a float64, utilizo el comando .astype()

amazon['actual_price'] = amazon['actual_price'].str.replace("₹",'').str.replace(",",'').astype('float64')

# para aplicar una función a cada fila, utilizo el comando .apply()
# utilizo lambda para crear una función.

exchange_rate = 0.012
amazon['discounted_price'] = amazon['discounted_price'].apply(lambda x: x * exchange_rate)
amazon['actual_price'] = amazon['actual_price'].apply(lambda x: x * exchange_rate)

# para redondear los precios a 2 decimales, utilizo el comando round()

amazon['discounted_price'] = round(amazon['discounted_price'], 2)
amazon['actual_price'] = round(amazon['actual_price'], 2)

# Creación de la columna discount_amount
 amazon['discount_amount'] = amazon['actual_price'] - amazon['discounted_price']

# para eliminar el % de la columna discount_percentage, utilizo el comando .str.replace()
# para convertir el tipo de dato en la columna discount_percentage a float64, utilizo el comando .astype()

amazon['discount_percentage'] = amazon['discount_percentage'].str.replace('%','').astype('float64')

# Divido discount_percentage por 100 para que los valores estén en decimales.

amazon['discount_percentage'] = amazon['discount_percentage'] / 100

# para ver los valores únicos en una columna y cuántos de cada valor hay, utilizo el comando .value_counts()

amazon['rating'].value_counts()

# para seleccionar la(s) fila(s) donde el valor de rating = |, utilizo el comando .loc[]

amazon.loc[amazon['rating'] == '|']

# para reemplazar | con 3.9 en la columna de calificación, utilizo el comando .str.replace()
# para convertir el tipo de dato en la columna de calificación a float64, utilizo el comando .astype()

amazon['rating'] = amazon['rating'].str.replace('|', '3.9').astype('float64')

# para eliminar las comas de la columna rating_count, utilizo el comando .str.replace()
# para convertir el tipo de dato en la columna rating_count a float64, utilizo el comando .astype()

amazon['rating_count'] = amazon['rating_count'].str.replace(',', '').astype('float64')

# para ver cuántos valores nulos hay en cada columna, utilizo el comando .isna() y .sum()

amazon.isna().sum()

# para encontrar el número de índice de las filas con 1 o más valores nulos, utilizo el comando .index
# para mostrar filas basadas en el número de índice, utilizoel comando .iloc

amazon.iloc[amazon[(amazon.isna().sum(axis = 1) >= 1)].index]

# para reemplazar los valores nulos con el valor promedio redondeado, utilizo el comando .fillna()
# para redondear el promedio al número entero más cercano, utilizo el comando round()
# para calcular el oromedio de rating_count, utilizo el comando .mean()

amazon['rating_count'] = amazon['rating_count'].fillna(round(amazon["rating_count"].mean()))

# para ver cuántos valores nulos hay en cada columna, utilizo el comando .isna().sum()

amazon.isna().sum()

# para eliminar filas duplicadas, utilizo el comando .drop_duplicates()

amazon = amazon.drop_duplicates()

# utilizo el comando .str.strip() para eliminar cualquier espacio en blanco al principio y al final de una cadena.

amazon['product_id'].str.strip()

# Creo un bucle for y una declaración if else para categorizar la escala de calificación.
# utilizo el comando .append() para agregar categorías de clasificación a una lista de clasificación.

ranking = []

for score in amazon['rating']:
    if score <= 0.9: ranking.append('Muy Malo')
    elif score <= 1.9: ranking.append('Malo')
    elif score <= 2.9: ranking.append('Promedio')
    elif score <= 3.9: ranking.append('Bueno')
    elif score <= 4.9: ranking.append('Muy Bueno')
    elif score == 5.0: ranking.append('Excelente')

# Agrego la lista de clasificación como una columna de clasificación en el dataframe de amazon.
# utilizo el comando .astype() para convertir los valores en la columna de clasificación en un tipo de dato de categoría.

amazon['ranking'] = ranking

amazon['ranking'] = amazon['ranking'].astype('category')