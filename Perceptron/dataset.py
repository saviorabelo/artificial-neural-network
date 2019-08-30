import random
import numpy as np
import pandas as pd

def artificialAND():
    # Data size 4*n*n
    n = 5
    x1 = [[random.uniform(0.2, 0.3), random.uniform(0.2, 0.3)] for _ in range(n) for _ in range(n)]
    x2 = [[random.uniform(0.2, 0.3), random.uniform(0.6, 0.7)] for _ in range(n) for _ in range(n)]
    x3 = [[random.uniform(0.6, 0.7), random.uniform(0.2, 0.3)] for _ in range(n) for _ in range(n)]
    x4 = [[random.uniform(0.6, 0.7), random.uniform(0.6, 0.7)] for _ in range(n) for _ in range(n)]

    x = np.concatenate((x1, x2), axis=0)
    x = np.concatenate((x, x3), axis=0)
    x = np.concatenate((x, x4), axis=0)

    y123 = [[0] for _ in range(3*n*n)]
    y4 = [[1] for _ in range(n*n)]
    y = np.concatenate((y123, y4), axis=0)

    return x, y

def irisFlower():
    dataset = pd.read_csv('Datasets/iris.data', sep=',')
    classe = dataset['Classe']
    x = dataset.drop(['Classe'], axis=1) 
    
    y = []
    for i in classe:
        if(i == 'Iris-setosa'):
            y.append([1, 0, 0])
        elif(i == 'Iris-versicolor'):
            y.append([0, 1, 0])
        elif(i == 'Iris-virginica'):
            y.append([0, 0, 1])
        else:
            print('Error!')

    return np.array(x), np.array(y)

def irisFlowerBinary(attribute):
    dataset = pd.read_csv('Datasets/iris.data', sep=',')
    classe = dataset['Classe']
    x = dataset.drop(['Classe'], axis=1)
    
    y = []
    for i in classe:
        if(i == attribute):
            y.append([1])
        else:
            y.append([0])

    return np.array(x), np.array(y)

def vertebralColumnBinary():
    dataset = pd.read_csv('Datasets/column_2C.dat', sep=' ')
    classe = dataset['Classe']
    x = dataset.drop(['Classe'], axis=1)

    y = []
    for i in classe:
        if(i == 'AB'):
            y.append([1])
        else:
            y.append([0])

    return np.array(x), np.array(y)

def vertebralColumn():
    dataset = pd.read_csv('Datasets/column_3C.dat', sep=' ')
    classe = dataset['Classe']
    x = dataset.drop(['Classe'], axis=1)
        
    y = []
    for i in classe:
        if(i == 'DH'):
            y.append([1, 0, 0])
        elif(i == 'SL'):
            y.append([0, 1, 0])
        elif(i == 'NO'):
            y.append([0, 0, 1])
        else:
            print('Error!')

    return np.array(x), np.array(y)
