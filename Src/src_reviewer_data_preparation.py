
import pandas as pd

# utilizo el comando .read_csv para importar el archivo .csv.
amazon = pd.read_csv(r'amazon.csv')

amazon_clean = pd.read_csv(r'/content/amazon_clean.csv')

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
reviewers.to_csv('reviewers_data.csv', index=False)