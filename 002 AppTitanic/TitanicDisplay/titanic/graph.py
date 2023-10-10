import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

def ruta():
    path = "./titanic/data/"
    data = pd.read_csv(path + 'submission_SVM.csv')
    return data

def get_graph_sex():
    data = ruta()
    survived_gender = data.groupby('Sex')['Survived'].mean() * 100

    fig, ax2 = plt.subplots()

    survived_gender.plot(kind='bar', color='lightsalmon')

    ax2.set_title('Supervivencia por género en el Titanic')
    ax2.set_xlabel('Género')
    ax2.set_ylabel('Tasa de supervivencia')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    labels = ['female' if sex == 0 else 'male' for sex in survived_gender.index]

    ax2.set_xticklabels(labels, rotation=0)

    plt.close()

    return fig


def get_graph_pclass():
    data = ruta()
    survived_class = data.groupby('Pclass')['Survived'].mean() * 100

    fig, ax2 = plt.subplots()

    survived_class.plot(kind='bar', colormap='Set3')

    ax2.set_title('Supervivencia por clase de cabina en el Titanic')
    ax2.set_xlabel('Clase de cabina')
    ax2.set_ylabel('Tasa de supervivencia')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.close()

    return fig


def get_graph_relatives():
    # Crear el gráfico
    data = ruta()
    survived_class = data.groupby('Relatives')['Survived'].mean() * 100

    fig, ax2 = plt.subplots()

    survived_class.plot(kind='bar', color='lightpink')

    ax2.set_title('Supervivencia por número de familiares')
    ax2.set_xlabel('Familiares')
    ax2.set_ylabel('Tasa de supervivencia')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.close()

    return fig


def get_graph_age():
    data = ruta()
    data['AgeGroup'] = data['Age'].apply(lambda age: f'{(int(age) // 10) * 10}-{(int(age) // 10 + 1) * 10 - 1}')

    survived_class = data.groupby('AgeGroup')['Survived'].mean() * 100

    fig, ax2 = plt.subplots()

    survived_class.plot(kind='bar', color='skyblue')

    ax2.set_title('Supervivencia por edad')
    ax2.set_xlabel('Edad')
    ax2.set_ylabel('Tasa de supervivencia')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.close()

    return fig
