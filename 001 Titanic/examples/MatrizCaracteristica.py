# 04/10/2023
# 05/10/2023

# Ejemplo Matriz caracteristica:
# Matriz numerica, transforma las 'categorias' a numeros,
# para que los algoritmos de Machine Learning (ML) puedan aprender

import pandas as pd

# diccionario data
data = {
    "Pclass": [1, 2, 3, 1, 2],
    "Sex": ["Male", "Female", "Female", "Male", "Female"]
}

# Creando el DF a partir del diccionario
train_data = pd.DataFrame(data)

# Creando la matriz caracteristica X a partir de features
features = ["Pclass", "Sex"]
X = pd.get_dummies(train_data[features])

print(X)