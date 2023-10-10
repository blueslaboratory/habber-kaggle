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
# importamos la libreria KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
# imputador para rellenar valores faltantes
from sklearn.impute import SimpleImputer
# validacion cruzada
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

print('\ntrain_data.Sex == \'female\'')
print(train_data.Sex == 'female')

print('\ntrain_data.loc[train_data.Sex == \'female\']')
print(train_data.loc[train_data.Sex == 'female'])

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



print('\n*** ML: K-Nearest Neighbours Algorithm (KNN) ***')

# y: variable objetivo a predecir aka target
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


# Creando el modelo KNN
# Aquí puedes ajustar el número de vecinos (neighbors) según tus necesidades
model = KNeighborsClassifier(n_neighbors=5)

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
output.to_csv('.\\output\\Submision.csv', index=False)

# Si llega hasta aqui todo deberia de haber funcionado
print("Your submission was successfully saved!")



print('\n*** SCORES: VALIDACION DEL MODELO CRUZADA ***')

# Definir las características (features)
# features sesgadas: accuracy de 0.99
# features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Survived"]
# features sin sesgar: accuracy de 0.81-0.82
features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']

# Crear las variables ficticias (dummies)
X_train_score = pd.get_dummies(train_data[features])
# print("\nX_train_score")
# print(X_train_score)

# Pasando la matriz X_train a un DataFrame
X_train_df = pd.DataFrame(X_train_score, columns=X_train_score.columns)
# print('\nX_train_df')
# print(X_train_df)


# Entrenas al modelo con los datos de X_train y luego lo testeas con los datos de train
# X_train_score solo funciona con datos numericos y no con datos categoricos
# Remueve la columna 'Survived' de X_train_df y guárdala en Y_train_score
Y_train_score = X_train_df.pop('Survived')
print('\nY_train_score')
print(Y_train_score)


# Escalar los datos: que todas las caracteristicas tengan la misma escala --> normalizacion entre [0,1]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_score)

# Nuevo modelo knn con neighbors = ?
knn_model = KNeighborsClassifier(n_neighbors=5)

'''
cv=5: Indica que se está realizando una validación cruzada con 5 divisiones (5-fold cross-validation). 
      Esto significa que el conjunto de entrenamiento se divide en 5 partes iguales, y se realiza el 
      entrenamiento y la evaluación del modelo 5 veces, utilizando una parte diferente como conjunto de 
      prueba en cada iteración.
scoring: hay varios tipos de scoring
         --> accuracy: cuando las clases del conjunto de datos no estan altamente desequilibradas
         --> roc_auc: si no se distingue bien entre clases positivas y negativas
'''
scores_accuracy = cross_val_score(knn_model, X_train_scaled, Y_train_score, cv=5, scoring='accuracy')
scores_roc_auc = cross_val_score(knn_model, X_train_scaled, Y_train_score, cv=5, scoring='roc_auc')

# Imprimir el score promedio
print()
print("Precisión media (scoring=accuracy):", scores_accuracy.mean())
print("Precisión media (scoring=roc_auc):", scores_roc_auc.mean())




print('\n*** CSV CON TODAS LAS COLUMNAS ***')
# Crear un DataFrame llamado output con todas las columnas de test_data
output = test_data.copy()

# Agregar la columna 'Survived' con las predicciones
output['Survived'] = predictions

# Guardar el output en un csv
# index=True crearia una columna adicional con ids autoincrementales
output.to_csv('.\\output\\SubmisionCompleteTable.csv', index=False)

print("CSV con todas las columnas was successfully saved!")