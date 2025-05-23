
import pandas as pd

# utilizo el comando .read_csv para importar el archivo .csv.
ruta_absoluta = '/Users/kaabil/Documents/EDA Amazon/Data/Raw/amazon.csv'
amazon = pd.read_csv(ruta_absoluta) 

# analizo las 5 primeras filas
amazon.head()

# para ver si hay algún valor extraño en el dataframe, Analizo las estadísticas descriptivas
amazon.describe()

# para mostrar qué tipos de datos hay en cada columna y cuántos valores no nulos hay, utilizo  el comando .info()
amazon.info()

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

# utilizo el comando .str.split() para dividir los valores de cadena en la columna de categoría usando | como delimitador.
# expand = True insertará cada elemento dividido en una columna separada.

splitcategory = amazon['category'].str.split('|', expand = True)
splitcategory

# utilizo el comando .isna().sum() para ver cuántos valores nulos hay en cada columna.

splitcategory.isna().sum()

# utilizo el comando .rename() para renombrar la columna 0 a "category" y la columna 1 a "subcategory".

splitcategory = splitcategory.rename(columns = {0:'category',
                                                1:'subcategory'})
splitcategory

# utilizo el comando .unique() para obtener una lista de valores únicos en la columna de categoría.

splitcategory['category'].unique()

# Formateo los valores de la categoría para que sean más fáciles de leer.
# Uso .str.replace() para agregar espacios a los valores de la categoría.

splitcategory['category'] = splitcategory['category'].str.replace('&',
                                                                  ' & ')

splitcategory['category'] = splitcategory['category'].str.replace('MusicalInstruments',
                                                                  'Musical Instruments')

splitcategory['category'] = splitcategory['category'].str.replace('OfficeProducts',
                                                                  'Office Products')

splitcategory['category'] = splitcategory['category'].str.replace('HomeImprovement',
                                                                  'Home Improvement')

# utilizoel comando .unique() para obtener una lista de valores únicos en la columna de subcategoría.

splitcategory['subcategory'].unique()

# utilizo .str.replace() para agregar espacios a los valores de la subcategoría.

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('&',
                                                                        ' & ')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace(',',
                                                                        ', ')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('NetworkingDevices',
                                                                        'Networking Devices')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('HomeTheater',
                                                                        'Home Theater')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('HomeAudio',
                                                                        'Home Audio')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('WearableTechnology',
                                                                        'Wearable Technology')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('ExternalDevices',
                                                                        'External Devices')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('DataStorage',
                                                                        'Data Storage')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('GeneralPurposeBatteries',
                                                                        'General Purpose Batteries')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('BatteryChargers',
                                                                        'Battery Chargers')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('OfficePaperProducts',
                                                                        'Office Paper Products')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('CraftMaterials',
                                                                        'Craft Materials')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('OfficeElectronics',
                                                                        'Office Electronics')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('PowerAccessories',
                                                                        'Power Accessories')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('HomeAppliances',
                                                                        'Home Appliances')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('AirQuality',
                                                                        'Air Quality')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('HomeStorage',
                                                                        'Home Storage')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('CarAccessories',
                                                                        'Car Accessories')

splitcategory['subcategory'] = splitcategory['subcategory'].str.replace('HomeMedicalSupplies',
                                                                        'Home Medical Supplies')

# utilizo el comando .drop() para eliminar la columna de categoría del dataframe de amazon.

amazon = amazon.drop(columns = 'category')

# Agrego la nueva columna de categoría y la columna de subcategoría al dataframe de amazon.
amazon['category'] = splitcategory['category']
amazon['subcategory'] = splitcategory['subcategory']

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

amazon_clean = amazon[['product_id',
                       'product_name',
                       'actual_price',
                       'discounted_price',
                       'discount_amount',
                       'discount_percentage',
                       'category',
                       'subcategory',
                       'rating',
                       'rating_count',
                       'ranking']]
amazon_clean

# Guardar el DataFrame limpio en un archivo CSV
amazon_clean.to_csv('../Data/Processed/amazon_clean.csv', index=False)

# utilizo el comando .str.split() para dividir los valores de cadena en las columnas user_id y user_name usando ',' como delimitador.
# expand = False insertará cada elemento dividido en una lista.

split_user_id = amazon['user_id'].str.split(',', expand = False)
split_user_name = amazon['user_name'].str.split(',', expand = False)

# utilizo el comando .explode() para dividir cada elemento de una lista en una fila.
# Nota: Aunque cada elemento de la lista está en una fila diferente, los elementos comparten el mismo número de índice.
# Ejemplo: Una lista con 5 valores se dividirá en 5 filas, pero cada una de esas 5 filas tendrá el mismo número de índice.

id_rows = split_user_id.explode()
name_rows = split_user_name.explode()

# utilizo el comando DataFrame() para crear un dataframe usando the exploded lists.

df_id_rows = pd.DataFrame(id_rows)
df_name_rows = pd.DataFrame(name_rows)

# Agrego las columnas product_name, category y subcategory del dataframe amazon_clean al dataframe df_name_rows.

df_name_rows['product_name'] = amazon_clean['product_name']
df_name_rows['category'] = amazon_clean['category']
df_name_rows['subcategory'] = amazon_clean['subcategory']

# utilizo el comando .reset_index() para restablecer el índice de modo que cada fila tenga su propio número de índice.

df_id_rows = df_id_rows.reset_index(drop = True)
df_name_rows = df_name_rows.reset_index(drop = True)

# utilizo el comando .merge() para fusionar 2 dataframes juntos.

reviewers = pd.merge(df_id_rows, df_name_rows, left_index = True, right_index = True)
reviewers

# utilizo el comando .isna().sum() para ver cuántos valores nulos hay en el dataframe reviewers.

reviewers.isna().sum()

# Guardar el DataFrame reviewers en un archivo CSV
reviewers.to_csv('../Data/Processed/reviewers_data.csv', index=False)

# Seleccionar las columnas necesarias para el modelo
columns_to_select = ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']
filtered_data = amazon_clean[columns_to_select]

# Definir las variables X e y
X = filtered_data[['actual_price', 'discount_percentage', 'rating', 'rating_count']]
y = filtered_data['discounted_price']

# Mostrar en pantalla las variables X e y de forma estructurada
print("Variables X (entrada):")


print("\nVariable y (objetivo):")


# Guardar el resultado en un nuevo archivo CSV para el modelo
filtered_data.to_csv('../Data/Processed/amazon_dataset_final.csv', index=False)