# 04/10/2023
# 05/10/2023


######################
# LINKS ##############
######################

# Kaggle competition
# https://www.kaggle.com/competitions/titanic

# Vamos a seguir el tutorial recomendado para empezar
# https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook

# Otros tutoriales que parecen interesantes
# https://www.kaggle.com/code/gusthema/titanic-competition-w-tensorflow-decision-forests
# https://www.kaggle.com/code/aagyapalkaur/top8-titanic-competition-eda-voting-classifier/notebook
# https://www.kaggle.com/code/urbanspr1nter/titanic-machine-learning-from-disaster/notebook

# Todos los algoritmos de ML
# https://www.kaggle.com/code/johnnydoyes/all-ml-algos

# YouTube: Beginner Data Science Portfolio Project Walkthrough (Kaggle Titanic)


###########################################
# INSTALAR LAS LIBRERIAS: DESDE LA TERMINAL
###########################################

# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install scikit-learn


#############################################
# GETTING STARTED WITH TITANIC ##############
#############################################

# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# importamos la libreria RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# imputador para rellenar valores faltantes
from sklearn.impute import SimpleImputer

import os

# ruta de los .csv
ruta = '.\\data'

print('*** RUTAS DE LOS CSV ***')

for dirname, _, filenames in os.walk(ruta):
    for filename in filenames:
        print(os.path.join(dirname, filename))



print('\n*** CREAR LOS DATAFRAME: LEER LOS CSV ***')

'''
PassengerId - Identificación del Pasajero
Pclass - Clase del Pasajero
Name - Nombre del Pasajero
Sex - Género del Pasajero
Age - Edad del Pasajero
SibSp - Número de Hermanos/Cónyuges a Bordo (Siblings/Spouses)
Parch - Número de Padres/Hijos a Bordo (Parents/Children)
Ticket - Número de Boleto
Fare - Tarifa Pagada por el Pasajero
Cabin - Cabina del Pasajero
Embarked - Puerto de Embarque del Pasajero
'''

ruta_test = ".\\data\\test.csv"
ruta_train = ".\\data\\train.csv"

# importa data de test.csv y train.csv: esto son los dataframes
test_data = pd.read_csv(ruta_test)
train_data = pd.read_csv(ruta_train)

# muestrame los 5 primeros registros
print(test_data.head())



print('\n*** MOSTRAR TODAS LAS COLUMNAS DEL DATAFRAME ***')

# Establecer la opción para mostrar todas las columnas sin truncar
pd.set_option('display.max_columns', None)

# Mostrar las primeras filas del DataFrame
print(test_data.head())


print('\n*** EXPLORANDO UN PATRON, DATOS Y ESTADISTICAS ***')

# Explorando un patron:
# --> Las mujeres viven
# --> Los hombres mueren

'''
train_data: dataframe (DF)
train_data.Sex == 'female': crea una serie booleana, 
                            si la columna Sex = female devuelve True
train_data.loc[]: filtra filas del DF basandose en la condicion de 
                  dentro de los corchetes --> son todas female
["Survived"]: se coloca despues de train_data.loc[] para especificar 
              que columna queremos seleccionar del DF resultante
              en este caso solo seleccionamos la columna "Survived" 
              de las filas que cumplan el genero femenino
'''

# devuelveme una columna con si es female (True) o no (False)
print('\ntrain_data.Sex == \'female\'')
print(train_data.Sex == 'female')

# devuelveme toda la tabla solo para los valores en los que es female == True
print('\ntrain_data.loc[train_data.Sex == \'female\']')
print(train_data.loc[train_data.Sex == 'female'])

# devuelveme solo la columna Survived del anterior DF
print('\ntrain_data.loc[train_data.Sex == \'female\'][\"Survived\"]')
print(train_data.loc[train_data.Sex == 'female']["Survived"])


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

missing_age_count = (train_data['Age'].isnull()).sum()
rate_missing_age = missing_age_count / len(train_data['Age'])


print('\nPorcentajes: ')
print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)
print("% of missing age values:", rate_missing_age)

print('\nTabla de nulos:')
print(train_data.isnull().sum())

'''
Since gender seems to be such a strong indicator of survival, 
the submission file in gender_submission.csv is not a bad first guess!

But at the end of the day, this gender-based submission bases its 
predictions on only a single column. As you can imagine, by considering 
multiple columns, we can discover more complex patterns

Since it is quite difficult to consider several columns at once 
(or, it would take a long time to consider all possible patterns in 
many different columns simultaneously), we'll use machine learning to 
automate this for us.
'''

print('\n*** IMPUTADOR PARA RELLENAR VALORES FALTANTES DE AGE ***')

# Reemplazar valores no numéricos con NaN en la columna 'Age'
# 'coerce' --> si no eres capaz de convertirmerlo a numerico ponme un NaN
train_data['Age'] = pd.to_numeric(train_data['Age'], errors='coerce')

# Crear un imputador que use la media
imputer = SimpleImputer(strategy='mean')

# Aplicar el imputador solo a la columna 'Age' de tus datos
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.fit_transform(test_data[['Age']])

print('\nTrain DataFrame rellenado con Imputer')
print(train_data)
print('\nTest DataFrame rellenado con Imputer')
print(test_data)

print('\nTabla de nulos:')
print(train_data.isnull().sum())



print('\n*** ML: RANDOM FOREST MODEL (RFC) ***')
# Your 1st Machine Learning (ML) model: Random Forest Model
# Formación\Ejercicios\ML & DL\images

# y: variable objetivo a predecir
# Contiene etiquetas de supervivencia del conjunto de datos de entrenamiento
# se asume que train data contiene la columna ["Survived"]
y = train_data["Survived"]
print('\ntrain_data[\"Survived\"]')
print(y)


# features: seleccion de las caracteristicas para entrenar el modelo
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

# Matrices de caracteristicas de train y test a traves de pandas
# Convierte las variables categoricas a variables numericas
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

print('\nMatriz caracteristica X_train')
print(X_train)
print('\nMatriz caracteristica X_test')
print(X_test)


# Creando el modelo RandomForestClassifier
'''
Hiperparametros del modelo:

--> n_estimators=100: nº de arboles de decision que se crearan en el bosques (100)
--> max_depth=5: profundidad de cada arbol
                 si tiene mucha profundidad hay un sobreajuste (no generaliza bien)
--> random_state=1: semilla para la generacion de nº pseudoaleatorios
                    garantiza que los resultados sean reproducibles.
                    Fija un punto de inicio para la generacion de numeros pseudoaleatorios.
                    --> Reproducibilidad: 
                        Misma semilla, mismos resultados
                    --> Consistencia: 
                        Todos debemos utilizar la misma semilla, mismos resultados
                    --> Depuracion y ajustes de hiperparametros:
                        Hiperparametros: configs antes de iniciar el proceso de entrenamiento.
                        Al ajustar el modelo es util que las diferencias de rendimiento no se
                        deban a variaciones aleatorias, sino a cambios en hiperparametros o datos.
                        La semilla asegura que la variabilidad debido a la aleatoriedad sea controlada
'''
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Entrenando el modelo con los datos de entrenamiento
# model.fit(matriz_caracteristica, variable_objetivo_a_predecir)
model.fit(X_train, y)

# A partir del modelo, con la matriz caracteristica X_test hazme las predicciones
predictions = model.predict(X_test)

# Crear un DataFrame llamado output con 2 columnas: PassengerId y Survived
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Guardar el output en un csv
# index=True crearia una columna adicional con ids autoincrementales
# RandomForestClassifier (RFC)
output.to_csv('.\\output\\RandomForestClassifierSubmision.csv', index=False)

# Si llega hasta aqui todo deberia de haber funcionado
print("Your submission was successfully saved!")



print('\n*** CSV CON TODAS LAS COLUMNAS ***')
# Crear un DataFrame llamado output con todas las columnas de test_data
output = test_data.copy()

# Agregar la columna 'Survived' con las predicciones
output['Survived'] = predictions

# Guardar el output en un csv
# index=True crearia una columna adicional con ids autoincrementales
output.to_csv('.\\output\\RandomForestClassifierSubmisionCompleteTable.csv', index=False)

print("CSV con todas las columnas was successfully saved!")