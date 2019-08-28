import numpy as np
import pandas as pd

def irisFlower():
    dataset = pd.read_csv('Datasets/iris.data', sep=',')
    flowers = np.array(dataset['Classe'])

    x = np.array(dataset.drop(['Classe'], axis=1))
    
    y = []
    for flower in flowers:
        if(flower == 'Iris-setosa'):
            y.append([1, 0, 0])
        elif(flower == 'Iris-versicolor'):
            y.append([0, 1, 0])
        elif(flower == 'Iris-virginica'):
            y.append([0, 0, 1])
        else:
            print('Error!')

    return np.array(x), np.array(y)

def irisFlowerBinary(attribute):
    dataset = pd.read_csv('Datasets/iris.data', sep=',')
    flowers = np.array(dataset['Classe'])

    x = np.array(dataset.drop(['Classe'], axis=1))
    
    y = []
    for flower in flowers:
        if(flower == attribute):
            y.append([1])
        else:
            y.append([0])

    return np.array(x), np.array(y)

