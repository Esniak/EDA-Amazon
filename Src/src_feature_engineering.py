
import pandas as pd

# utilizo el comando .read_csv para importar el archivo .csv.
amazon = pd.read_csv(r'amazon.csv')

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